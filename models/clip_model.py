import torch
from torch import nn
import torch.nn.functional as F


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


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, bilinear: bool = True, base_c: int = 32):
        super(ImageEncoder, self).__init__()
        self.in_channels = in_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8 // factor)

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

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # 在通道维度上进行平均池化
        channel_avg_pool = torch.mean(x4, dim=1)  # 沿着通道维度进行平均池化
        # 将平均池化后的特征张量展平为一维向量
        flattened_feature = torch.flatten(channel_avg_pool, start_dim=1)  # 从第1维开始展平
        return flattened_feature


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


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim=256):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class CLIPModel(nn.Module):
    def __init__(self, temperature=1.0, image_embedding=4096, text_embedding=768):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder(text_embedding)
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, image, text):
        # Getting Image and Text Features
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text)
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax((images_similarity + texts_similarity) / 2 * self.temperature, dim=-1)
        texts_loss = cross_entropy(logits, targets, reduction="none")
        images_loss = cross_entropy(logits.T, targets.T, reduction="none")
        loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction="none"):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
