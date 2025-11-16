import torch
from models import dcgan
import torchio as tio
import numpy as np
from models.clip_model import CLIPModel

# from torchio.transforms import RandomAffine, RandomNoise


transform = tio.Compose(
    [
        tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
        tio.RandomNoise(p=0.5),
    ]
)


def get_subject(folder_name, model_str, blood=None, blood1=None, label=None):
    T1fold = folder_name + "/T1.nii.gz"
    maskfold = folder_name + "/mask.nii.gz"
    PETfold = folder_name + "/pet.nii.gz"
    # atlasfold = image_name + "/atlas.nii"
    T1 = tio.ScalarImage(T1fold)
    pet = tio.ScalarImage(PETfold)
    mask = tio.LabelMap(maskfold)
    label_tensor = torch.tensor(int(label), dtype=torch.long)
    if model_str == "vision":
        subject = tio.Subject(T1=T1, mask=mask, pet=pet, label=label_tensor)
    else:
        subject = tio.Subject(
            T1=T1,
            mask=mask,
            pet=pet,
            blood=torch.from_numpy(blood),
            blood1=torch.from_numpy(blood1),
            label=label_tensor,
        )
        # subject = tio.Subject(T1=T1, mask=mask, pet=pet, blood=torch.from_numpy(blood), label=label_tensor)

    return subject


def build_dataloader(df, batch_size, num_workers, img_path, model_args=None, sets="val"):
    images = [img_path + i for i in df["ID"].tolist()]
    model_str = model_args["model"]
    if model_str == "vision":
        bloods = [None] * len(df)
        bloods1 = [None] * len(df)
    elif model_str == "num":
        blood_str = [
            "Aβ40 (pg/mL)",
            "Aβ42(pg/mL)",
            "T-Tau(pg/mL)",
            "P-Tau 181(pg/mL)",
            "Aβ42/40",
            "Nfl(pg/mL)",
            "P-Tau 181/Aβ42",
        ]
        bloods = df[blood_str].values
        # 找到每列的最小值和最大值
        # min_vals = bloods.min(axis=0)
        # max_vals = bloods.max(axis=0)
        # scaled_bloods = (bloods - min_vals) / (max_vals - min_vals)
        scaled_bloods = bloods
        bloods = [scaled_bloods[i, :].astype(np.float32) for i in range(len(bloods))]
        bloods1 = [None] * len(df)
    else:
        if sets == "gen":
            bloods_ld = np.load(model_args["text_embedding"])
        else:
            bloods_ld = np.load(model_args["text_embedding_val"])
            # bloods_ld = np.load("data/HS/LLM/cutoff.npz")
            # bloods_ld = np.load("data/HS/LLM/wp_b.npz")
            # bloods_ld = np.load("data/HS/LLM/wp.npz")
            # bloods_ld = np.load("data/HS/LLM/onlyp_b.npz")

        bloods_ld1 = np.load(model_args["text_embedding_D"])
        # bloods_ld1 = bloods_ld
        bloods = [bloods_ld[ids] for ids in df["ID"].tolist()]
        bloods1 = [bloods_ld1[ids] for ids in df["ID"].tolist()]
    labels = df["labels"].tolist()
    subjects = []
    for idx, image in enumerate(images):
        subject = get_subject(image, model_str, bloods[idx], bloods1[idx], labels[idx])
        subjects.append(subject)
    if sets == "train":
        subjects_dataset = tio.SubjectsDataset(subjects, transform=transform)
    else:
        subjects_dataset = tio.SubjectsDataset(subjects)
    dataloader = torch.utils.data.DataLoader(
        dataset=subjects_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=False
    )
    return dataloader


def build_g_d(model_args, encoder_pth=None, input_size=128, text_pth=None):
    model_str = model_args["model"]
    # Initialize generator and discriminator
    if model_str == "vision":
        generator = dcgan.UNet()
    else:
        if model_str == "num":
            control_size = 7
        elif model_str == "gpt":
            control_size = 2560
        elif model_str == "bert":
            control_size = 768
        elif model_str == "clip":
            control_size = 512
        else:
            raise ValueError("model_str error")
        generator = dcgan.UNet_control_affine(control_size=control_size)
        # generator = dcgan.UNet_control_multi_affine(control_size=control_size)

    discriminator = dcgan.Discriminator(input_size=input_size)
    # discriminator = dcgan.DiscriminatorC(input_size=input_size)
    # discriminator = dcgan.PatchDiscriminator(input_size=input_size)
    # Initialize weights
    generator.apply(dcgan.weights_init_normal)
    discriminator.apply(dcgan.weights_init_normal)
    if model_str != "vision":
        saved_params = torch.load(encoder_pth)
        matched_params = {key: saved_params[key] for key in saved_params.keys() if key in generator.state_dict().keys()}
        for key in matched_params.keys():
            if ("down" in key) | ("in_conv" in key):  # 如果键名中包含 'down'，则加载参数
                print(key)
                generator.state_dict()[key].copy_(matched_params[key])
    return generator, discriminator


def load_g(model_args, pth_path):
    model_str = model_args["model"]
    if model_str == "vision":
        generator = dcgan.UNet()
    else:
        if model_str == "num":
            control_size = 7
        elif model_str == "gpt":
            control_size = 2560
        elif model_str == "bert":
            control_size = 768
        elif model_str == "clip":
            control_size = 512
        else:
            raise ValueError("model_str error")
        generator = dcgan.UNet_control_affine(control_size=control_size)

    generator.load_state_dict(torch.load(pth_path))
    generator.cuda()
    return generator


def build_clip():
    model = CLIPModel()
    return model
