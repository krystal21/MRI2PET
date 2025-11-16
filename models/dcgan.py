import torch.nn as nn

# import torch.nn.functional as F
import torch
from torchsummary import summary


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv3d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm3d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(nn.MaxPool3d(2, stride=2), DoubleConv(in_channels, out_channels))


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, bilinear: bool = True, base_c: int = 32):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8 // factor)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = nn.Conv3d(base_c, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out_conv(x)
        return x


class Affine(nn.Module):
    def __init__(self, size_in, size_out):
        super(Affine, self).__init__()
        self.control_scale = nn.Sequential(
            nn.Linear(size_in, size_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(size_out, size_out),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.control_shift = nn.Sequential(
            nn.Linear(size_in, size_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(size_out, size_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, c):
        scale = self.control_scale(c)
        shift = self.control_shift(c)
        y = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x + shift.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return y + x


class Cat(nn.Module):
    def __init__(self, size_in, channel_out, image_channel):
        super(Cat, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(size_in, channel_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channel_out, channel_out),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.ad_conv = nn.Conv3d(channel_out + image_channel, image_channel, kernel_size=1)

    def forward(self, x, c):
        s3 = x.shape[3]
        ch_c = self.mlp(c)
        ch_c = ch_c.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        ch_c = ch_c.repeat(1, 1, s3, s3, s3)
        cat = torch.cat((x, ch_c), dim=1)
        x = self.ad_conv(cat)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, out_dim):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, out_dim)

    def forward(self, inputq, inputk):
        batch_size = inputq.size(0)
        seq_q = inputq.size(1)
        seq_k = inputk.size(1)

        # Linear transformations
        query = self.query_linear(inputq)
        key = self.key_linear(inputk)
        value = self.value_linear(inputk)

        # Reshape for multi-head attention
        query = query.view(batch_size, seq_q, self.num_heads, self.d_k)
        key = key.view(batch_size, seq_k, self.num_heads, self.d_k)
        value = value.view(batch_size, seq_k, self.num_heads, self.d_k)

        # Transpose dimensions for matrix multiplication
        query = query.transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        key = key.transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        value = value.transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1))  # [batch_size, num_heads, seq_len, seq_len]
        scores = scores / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        attention_weights = nn.Softmax(dim=-1)(scores)
        attention_output = torch.matmul(attention_weights, value)  # [batch_size, num_heads, seq_len, d_k]

        # Concatenate and reshape
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_q, self.d_model)

        # Linear transformation for final output
        attention_output = self.output_linear(attention_output)

        return attention_output


class CrossAttention(nn.Module):
    def __init__(self, text_dim, image_dim, hidden_dim):
        super(CrossAttention, self).__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim

        self.text_linear = nn.Linear(text_dim, hidden_dim)
        self.image_linear = nn.Linear(image_dim, hidden_dim)

        self.attention = MultiHeadAttention(hidden_dim, 8, image_dim)
        self.proj_out = nn.Conv3d(1, 128, kernel_size=1, stride=1, padding=0)

    def forward(self, image_input, text_input):
        b, c, h, w, z = image_input.shape
        image_input = image_input.view(b, c, h * w * z)
        text_input = text_input.unsqueeze(1)
        text_proj = self.text_linear(text_input)  # (b, 1, hidden_dim)
        image_proj = self.image_linear(image_input)  # (b, 128, hidden_dim)
        qkv = self.attention(image_proj, text_proj)
        output = qkv.view(b, c, h, w, z)
        # qkv = self.attention(text_proj, image_proj)
        # output = qkv.view(b, 1, h, w, z)
        # output = self.proj_out(output)
        return output


class UNet_control_affine(nn.Module):
    def __init__(self, in_channels=1, bilinear=True, base_c=32, control_size=7):
        super(UNet_control_affine, self).__init__()
        self.in_channels = in_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8 // factor)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = nn.Conv3d(base_c, 1, kernel_size=1)

        self.affine1 = Affine(control_size, base_c * 4)
        self._freeze_downsampling()

    def _freeze_downsampling(self):
        for param in self.in_conv.parameters():
            param.requires_grad = False
        for param in self.down1.parameters():
            param.requires_grad = False
        for param in self.down2.parameters():
            param.requires_grad = False
        for param in self.down3.parameters():
            param.requires_grad = False

    def forward(self, x, c):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.affine1(x4, c)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out_conv(x)
        return x


class UNet_control_multi_affine(nn.Module):
    def __init__(self, in_channels=1, bilinear=True, base_c=32, control_size=7):
        super(UNet_control_multi_affine, self).__init__()
        self.in_channels = in_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8 // factor)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = nn.Conv3d(base_c, 1, kernel_size=1)

        self.affine1 = Affine(control_size, base_c * 4)
        self.affine2 = Affine(control_size, base_c * 2)
        self.affine3 = Affine(control_size, base_c * 1)
        self._freeze_downsampling()

    def _freeze_downsampling(self):
        for param in self.in_conv.parameters():
            param.requires_grad = False
        for param in self.down1.parameters():
            param.requires_grad = False
        for param in self.down2.parameters():
            param.requires_grad = False
        for param in self.down3.parameters():
            param.requires_grad = False

    def forward(self, x, c):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.affine1(x4, c)
        x = self.up2(x4, x3)
        x = self.affine2(x, c)
        x = self.up3(x, x2)
        x = self.affine3(x, c)
        x = self.up4(x, x1)
        x = self.out_conv(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, text_dim):
        super().__init__()
        self.fc1 = nn.Linear(text_dim, text_dim)
        self.gelu1 = nn.GELU()
        self.fc2 = nn.Linear(text_dim, text_dim)
        self.gelu2 = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu1(x)
        x = self.fc2(x)
        x = self.gelu2(x)
        x = self.dropout(x)
        return x


class UNet_Control_CLIP(nn.Module):
    def __init__(self, in_channels=1, bilinear=True, base_c=32, control_size=7):
        super(UNet_Control_CLIP, self).__init__()
        self.in_channels = in_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8 // factor)
        # self.down4 = Down(base_c * 8, base_c * 16 // factor)
        # self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = nn.Conv3d(base_c, 1, kernel_size=1)

        self.pre_text = TextEncoder(control_size)
        self.affine = Affine(control_size, base_c * 4)
        # self.cat = Cat(control_size, base_c // 8, base_c * 4)

        # 将下采样部分的参数设置为不可训练
        self._freeze_downsampling()

    def _freeze_downsampling(self):
        for param in self.in_conv.parameters():
            param.requires_grad = False
        for param in self.down1.parameters():
            param.requires_grad = False
        for param in self.down2.parameters():
            param.requires_grad = False
        for param in self.down3.parameters():
            param.requires_grad = False

    def forward(self, x, c):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        c = self.pre_text(c)
        x4 = self.affine(x4, c)
        # x4 = self.cat(x4, c)

        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out_conv(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, channels=1, input_size=128):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters):
            block = [
                nn.Conv3d(in_filters, out_filters, 3, 2, 1),
                nn.BatchNorm3d(out_filters),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout3d(0.2),
            ]
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        ds_size = input_size // 2**4
        self.adv_layer = nn.Linear(128 * ds_size**3, 1)

    # def forward(self, img, condition_img):
    def forward(self, img):
        # img_cat = torch.cat((img, condition_img), dim=1)
        # out = self.model(img_cat)
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


class DiscriminatorC(nn.Module):
    def __init__(self, channels=2, input_size=128):
        super(DiscriminatorC, self).__init__()

        def discriminator_block(in_filters, out_filters):
            block = [
                nn.Conv3d(in_filters, out_filters, 3, 2, 1),
                nn.BatchNorm3d(out_filters),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout3d(0.2),
            ]
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        ds_size = input_size // 2**4
        self.adv_layer = nn.Linear(128 * ds_size**3, 1)
        self.mlp = nn.Linear(768, 32)
        # self.mlp = nn.Linear(7, 32)
        self.jointConv = nn.Sequential(
            nn.Conv3d(128 + 32, 128, 3, 1, 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, img, condition_img, text):
        # def forward(self, img, text):
        img_cat = torch.cat((img, condition_img), dim=1)
        x_code = self.model(img_cat)
        # x_code = self.model(img)
        c_code = self.mlp(text)
        c_code = c_code.view(-1, 32, 1, 1, 1)
        c_code = c_code.repeat(1, 1, 8, 8, 8)
        h_c_code = torch.cat((c_code, x_code), 1)
        out = self.jointConv(h_c_code)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


class PatchDiscriminator(nn.Module):
    def __init__(self, channels=2, input_size=128):
        super(PatchDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv3d(in_filters, out_filters, 3, 2, 1)]
            if normalize:
                layers.append(nn.BatchNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout3d(0.2))
            return layers

        layers = []

        layers.extend(discriminator_block(channels, 16, normalize=False))
        layers.extend(discriminator_block(16, 32))
        layers.extend(discriminator_block(32, 64))
        layers.extend(discriminator_block(64, 128))

        self.model = nn.Sequential(*layers)
        self.adv_layer = nn.Conv3d(128, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, img, condition_img):
        img_cat = torch.cat((img, condition_img), dim=1)
        features = self.model(img_cat)
        validity = self.adv_layer(features).view(-1, 1, 8, 8, 8)

        return validity


if __name__ == "__main__":
    # 创建模型实例
    # model = UNet().cuda()
    model = Discriminator().cuda()
    # model = Generator_control_BN().cuda()
    # model = Generator().cuda()
    # 打印模型结构摘要信息
    summary(model, (1, 128, 128, 128))
    # summary(model, (1, 128, 128, 128), batch_size=1)
