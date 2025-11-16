"""2024.1.3"""

import os
import logging
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.backends.cudnn as cudnn
from utils.trainer import train_cls, val_cls
from config.setting import get_config
import torchio as tio
from models.resnet3d import resnet18, resnet34, resnet50

# import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


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


transform = tio.Compose(
    [
        tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
        tio.RandomNoise(p=0.5),
    ]
)


def get_subject(folder_name, label=None):
    # maskfold = folder_name + "/mask.nii.gz"
    # PETfold = folder_name + "/T1.nii.gz"
    PETfold = folder_name + "pet.nii.gz"
    # print(PETfold)
    pet = tio.ScalarImage(PETfold)
    # mask = tio.LabelMap(maskfold)
    label_tensor = torch.tensor(int(label), dtype=torch.long)
    subject = tio.Subject(pet=pet, label=label_tensor)
    # print(subject["pet"][tio.DATA].shape)
    # print(subject["label"])
    return subject


def build_dataloader(df, batch_size, num_workers, img_path, sets="val"):
    images = [img_path + i for i in df["ID"].tolist()]
    # print(images)
    labels = df["labels"].tolist()
    subjects = []
    for idx, image in enumerate(images):
        subject = get_subject(image, labels[idx])
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
    # csv_path = "data/HS/fold_label3"
    csv_path = "data/HS/fold"
    # csv_path = "data/HS/fold_test"
    # img_path = args.path[args.model["dataset"]]["img_path"]
    # img_path = "gen_pet/wp_braw/"
    # img_path = "gen_pet/wopwp/"
    # img_path = "gen_pet/wpbwp/"
    # img_path = "gen_pet/wp/"
    # img_path = "gen_pet/cutoff/"
    img_path = "gen_pet/wp_b/"
    # img_path = "gen_pet/vision"
    # img_path = "/ailab/user/zhangzhengjie/PET/preprocessed/128_01/"

    # 保存路径
    pth_path = args.path["pth"]
    result_path = args.path["result"]
    # 训练参数
    batch_size, num_workers, epochs, save_freq = (
        args.training["batch_size"],
        args.training["num_workers"],
        args.training["epochs"],
        args.training["save_freq"],
    )
    # lr, weight_decay = 1e-3, 5e-4
    lr, weight_decay = 1e-4, 5e-4
    T_max, eta_min = 70, 1e-7
    batch_size = 16
    epochs = 70

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

            # dataloader_train = build_dataloader(df_train, batch_size, num_workers, img_path, "train")
            dataloader_train = build_dataloader(df_train, batch_size, num_workers, img_path, "x")
            dataloader_val = build_dataloader(df_val, batch_size, num_workers, img_path, "val")

            model = generate_model(model_depth=18, nb_class=2)
            cls_pth = "exp/cls/new/wpb-2024-05-07 04-07/pth/"
            cls_paths = {0: "0epoch55.pth", 1: "1epoch24.pth", 2: "2epoch4.pth", 3: "3epoch41.pth", 4: "4epoch1.pth"}
            model.load_state_dict(torch.load(f"{cls_pth}{cls_paths[fold]}"))
            # print(f"{ppp}/pth/{fold}epoch{idxmax}.pth")
            # Loss function
            n1 = df_train["labels"].sum()
            n0 = len(df_train) - df_train["labels"].sum()
            ratio = n0 / n1
            # ratio = 2.0
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

                losses_train, auc_train, acc_train, recall_train, spe_train = train_cls(
                    dataloader_train, model, optimizer, criterion
                )
                losses_val, auc_val, acc_val, recall_val, spe_val = val_cls(dataloader_val, model, criterion)

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
