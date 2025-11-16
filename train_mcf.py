"""2024.1.3"""

import os
import logging
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.backends.cudnn as cudnn
from utils.trainer import train_mcf, val_mcf
from config.setting import get_config
import torchio as tio
from models.resnet3d import resnet18, resnet34, resnet50
from sklearn.preprocessing import MinMaxScaler


def generate_model(model_depth=50, input_W=128, input_H=128, input_D=128, nb_class=2, input_channel=1):
    if model_depth == 18:
        model = resnet18(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type="A",
            num_seg_classes=1,
        )
        fc_input = 512
    elif model_depth == 34:
        model = resnet34(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type="A",
            num_seg_classes=1,
        )
        fc_input = 512
    elif model_depth == 50:
        model = resnet50(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type="B",
            num_seg_classes=1,
        )
        fc_input = 2048
    model.conv_seg = nn.Sequential(
        nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(), nn.Linear(in_features=fc_input, out_features=nb_class, bias=True)
    )
    # change layer segmentation to dense layer
    return model


class ANN(nn.Module):
    def __init__(self, input_size, output_size):
        super(ANN, self).__init__()
        self.middle = 64
        self.fc1 = nn.Linear(input_size, self.middle)
        self.fc2 = nn.Linear(self.middle, self.middle)
        self.fc3 = nn.Linear(self.middle, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ResNetEnsemble(nn.Module):
    def __init__(self, num_classes=2, model_depth=18):
        super(ResNetEnsemble, self).__init__()
        self.resnet18 = generate_model(model_depth=model_depth, nb_class=num_classes)
        self.ann = ANN(11, 2)

    def forward(self, x, y):
        # 分别传递输入到ResNet18和ResNet50模型中
        output1 = self.resnet18(x)
        output2 = self.ann(y)
        # output = (output1 + output2) / 2
        return output1, output2


transform = tio.Compose(
    [
        tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
        tio.RandomNoise(p=0.5),
    ]
)


def get_subject(folder_name, blood, label=None):
    # maskfold = folder_name + "/mask.nii.gz"
    # PETfold = folder_name + "/T1.nii.gz"
    PETfold = folder_name + "pet.nii.gz"
    pet = tio.ScalarImage(PETfold)
    # mask = tio.LabelMap(maskfold)
    label_tensor = torch.tensor(int(label), dtype=torch.long)
    subject = tio.Subject(pet=pet, label=label_tensor, blood=torch.from_numpy(blood).float())
    return subject


def build_dataloader(df, X, batch_size, num_workers, img_path, sets="val"):
    images = [img_path + i for i in df["ID"].tolist()]
    labels = df["labels"].tolist()
    subjects = []
    for idx, image in enumerate(images):
        subject = get_subject(image, X[idx], labels[idx])
        subjects.append(subject)
    if sets == "train":
        subjects_dataset = tio.SubjectsDataset(subjects, transform=transform)
    else:
        subjects_dataset = tio.SubjectsDataset(subjects)
    dataloader = torch.utils.data.DataLoader(
        dataset=subjects_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=False
    )
    return dataloader


if __name__ == "__main__":
    # 加载参数
    args, writer = get_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.training["gpu_id"]
    torch.manual_seed(args.training["seed"])
    # 读取路径
    csv_path = "data/HS/fold"
    # img_path = "result/vis/"
    # img_path = "gen_pet/cutoff/"
    # img_path = "gen_pet/wp_braw/"
    img_path = "gen_pet/wp_b/"
    # img_path = "/ailab/user/zhangzhengjie/PET/preprocessed/128_01/"

    # 保存路径
    pth_path = args.path["pth"]
    result_path = args.path["result"]
    batch_size, num_workers, epochs, save_freq = (
        args.training["batch_size"],
        args.training["num_workers"],
        args.training["epochs"],
        args.training["save_freq"],
    )
    lr, weight_decay = 1e-4, 5e-4
    T_max, eta_min = 70, 1e-6
    batch_size = 16
    epochs = 70
    pths = [
        "exp/cls/newn/wpb1-2024-06-27 22-33/pth/0epoch55.pth",
        "exp/cls/newn/wpb1-2024-06-27 22-33/pth/1epoch6.pth",
        "exp/cls/newn/wpb1-2024-06-27 22-33/pth/2epoch27.pth",
        "exp/cls/newn/wpb1-2024-06-27 22-33/pth/3epoch19.pth",
        "exp/cls/newn/wpb1-2024-06-27 22-33/pth/4epoch39.pth",
    ]

    for fold in range(5):
        if fold >= 0:
            logging.info("---------------fold: %s---------------", fold)
            # dataloader
            df_train = pd.read_csv(f"{csv_path}/{fold}train.csv")
            df_val = pd.read_csv(f"{csv_path}/{fold}val.csv")
            df_train = df_train[df_train["labels"] != 2]
            df_val = df_val[df_val["labels"] != 2]
            # df_train.loc[df_train["labels"] == 2, "labels"] = 1
            # df_val.loc[df_val["labels"] == 2, "labels"] = 1

            blood_str = [
                "Aβ40 (pg/mL)",
                "Aβ42(pg/mL)",
                "T-Tau(pg/mL)",
                "P-Tau 181(pg/mL)",
                "Aβ42/40",
                "Nfl(pg/mL)",
                "P-Tau 181/Aβ42",
                "Age",
                "MMSE",
                "Edu yrs",
                "MoCA-B",
            ]
            repnan = 0
            df_train = df_train.replace({"^空$": repnan, "^未做$": repnan, "^N$": repnan, "^999$": repnan}, regex=True)
            df_val = df_val.replace({"^空$": repnan, "^未做$": repnan, "^N$": repnan, "^999$": repnan}, regex=True)
            pad_str = "Edu yrs"
            df_train[pad_str] = df_train[pad_str].fillna(0)
            df_val[pad_str] = df_val[pad_str].fillna(0)
            pad_str = "MoCA-B"
            df_train[pad_str] = df_train[pad_str].fillna(0)
            df_val[pad_str] = df_val[pad_str].fillna(0)
            # blood_str = ["Nfl(pg/mL)", "P-Tau 181/Aβ42"]
            X_train = df_train[blood_str].values
            Y_train = df_train["labels"]
            X_val = df_val[blood_str].values
            Y_val = df_val["labels"]
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            dataloader_train = build_dataloader(df_train, X_train_scaled, batch_size, num_workers, img_path, "train")
            dataloader_val = build_dataloader(df_val, X_val_scaled, batch_size, num_workers, img_path, "val")

            # model
            # dfx = pd.read_csv(f"{ppp}/result/{fold}.csv")
            # dfx["z"] = dfx["spe_val"] + dfx["recall_val"] + dfx["auc_val"]
            # idxmax = dfx["z"].idxmax()
            # idxmax = dfx["auc_val"].idxmax()
            # res_pth = f"{ppp}/pth/{fold}epoch{idxmax}.pth"
            # print(res_pth)
            res_pth = pths[fold]
            blood_pth = f"exp/blood/new/{fold}best.pth"
            model = ResNetEnsemble()
            model.ann.load_state_dict(torch.load(blood_pth))
            model.resnet18.load_state_dict(torch.load(res_pth))

            # Loss function
            n1 = df_train["labels"].sum()
            n0 = len(df_train) - df_train["labels"].sum()
            # ratio = n0 / n1
            ratio = 2.0
            print(n0 / n1)
            weight = torch.tensor([1.0, float(ratio)])
            criterion = nn.CrossEntropyLoss(weight=weight)

            if torch.cuda.is_available():
                model.cuda()
                criterion.cuda()
                cudnn.benchmark = True

            # Optimizers
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

            # train
            list_results = []
            best = 0
            best_auc = 0
            for epoch in range(epochs):
                logging.info("epoch: %s", epoch)

                losses_train, auc_train, acc_train, recall_train, spe_train = train_mcf(
                    dataloader_train, model, optimizer, criterion
                )
                losses_val, auc_val, acc_val, recall_val, spe_val = val_mcf(dataloader_val, model, criterion)

                scheduler.step()

                writer.add_scalar(f"{fold}/train/loss", losses_train, epoch)
                writer.add_scalar(f"{fold}/val/loss", losses_val, epoch)
                writer.add_scalar(f"{fold}/train/acc", acc_train, epoch)
                writer.add_scalar(f"{fold}/val/acc", acc_val, epoch)
                writer.add_scalar(f"{fold}/train/auc", auc_train, epoch)
                writer.add_scalar(f"{fold}/val/auc", auc_val, epoch)
                writer.add_scalar(f"{fold}/train/recall", recall_train, epoch)
                writer.add_scalar(f"{fold}/val/recall", recall_val, epoch)
                writer.add_scalar(f"{fold}/train/pre", spe_train, epoch)
                writer.add_scalar(f"{fold}/val/pre", spe_val, epoch)
                # if acc_val >= best:
                #     torch.save(model.state_dict(), pth_path + "/" + str(fold) + "epoch" + str(epoch) + ".pth")
                #     best = acc_val
                # if auc_val >= best_auc:
                torch.save(model.state_dict(), pth_path + "/" + str(fold) + "epoch" + str(epoch) + ".pth")
                # best_auc = auc_val
                results = [auc_val, acc_val, recall_val, spe_val]
                list_results.append(results)
                df = pd.DataFrame(
                    {
                        "auc_val": list(np.array(list_results).T[0]),
                        "acc_val": list(np.array(list_results).T[1]),
                        "recall_val": list(np.array(list_results).T[2]),
                        "spe_val": list(np.array(list_results).T[3]),
                    }
                )
                df.to_csv(result_path + "/" + str(fold) + ".csv")
