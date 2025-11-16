"""2024.1.3"""

import os
import logging
import pandas as pd
import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.backends.cudnn as cudnn
from utils.builder import build_dataloader, build_g_d
from utils.trainer import train_step, val_step, train_step_text, val_step_text
from config.setting import get_config


if __name__ == "__main__":
    # model_args
    args, writer = get_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.training["gpu_id"]
    torch.manual_seed(args.training["seed"])
    model_args = args.model
    pth_list = args.pth_list
    # path
    csv_path = args.path[args.model["dataset"]]["csv_path"]
    img_path = args.path[args.model["dataset"]]["img_path"]
    pth_path, result_path = args.path["pth"], args.path["result"]
    # parameters
    batch_size, num_workers = args.training["batch_size"], args.training["num_workers"]
    epochs, save_freq = args.training["epochs"], args.training["save_freq"]
    lr_G, lr_D = args.optimizer["lr_G"], args.optimizer["lr_D"]
    momentum, weight_decay = args.optimizer["momentum"], args.optimizer["weight_decay"]
    b1, b2 = args.optimizer["b1"], args.optimizer["b2"]
    T_max, eta_min = args.scheduler["T_max"], args.scheduler["eta_min"]

    for fold in range(5):
        if fold <= 10:
            logging.info("---------------fold: %s---------------", fold)
            # dataloader
            df_train = pd.read_csv(f"{csv_path}/{fold}train.csv")
            df_val = pd.read_csv(f"{csv_path}/{fold}val.csv")
            df_train = df_train[df_train["labels"] != 2]
            df_val = df_val[df_val["labels"] != 2]
            # df_train.loc[df_train["labels"] == 2, "labels"] = 1
            # df_val.loc[df_val["labels"] == 2, "labels"] = 1
            dataloader_train = build_dataloader(df_train, batch_size, num_workers, img_path, model_args, sets="gen")
            dataloader_val = build_dataloader(df_val, batch_size, num_workers, img_path, model_args)

            # model
            generator, discriminator = build_g_d(model_args, pth_list[fold])

            # Loss function
            adversarial_loss = torch.nn.BCEWithLogitsLoss()
            error_loss = torch.nn.MSELoss()

            if torch.cuda.is_available():
                generator.cuda()
                discriminator.cuda()
                adversarial_loss.cuda()
                error_loss.cuda()
                cudnn.benchmark = True

            # Optimizers
            optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_G, betas=(b1, b2), weight_decay=weight_decay)
            optimizer_D = torch.optim.Adam(
                discriminator.parameters(), lr=lr_D, betas=(b1, b2), weight_decay=weight_decay
            )
            # scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=7, verbose=True)
            scheduler_D = CosineAnnealingLR(optimizer_D, T_max=T_max, eta_min=eta_min)
            scheduler_G = CosineAnnealingLR(optimizer_G, T_max=T_max, eta_min=eta_min)

            # train
            list_results = []
            for epoch in range(epochs):
                logging.info("epoch: %s", epoch)
                if model_args["model"] == "vision":
                    losses_G, losses_D, losses_BCE, losses_MSE, psnr_train = train_step(
                        dataloader_train,
                        generator,
                        discriminator,
                        optimizer_D,
                        optimizer_G,
                        adversarial_loss,
                        error_loss,
                    )
                    losses_MSE_val, psnr_val = val_step(dataloader_val, generator, error_loss)
                else:
                    losses_G, losses_D, losses_BCE, losses_MSE, psnr_train = train_step_text(
                        dataloader_train,
                        generator,
                        discriminator,
                        optimizer_D,
                        optimizer_G,
                        adversarial_loss,
                        error_loss,
                    )
                    losses_MSE_val, psnr_val = val_step_text(dataloader_val, generator, error_loss)

                scheduler_D.step()
                scheduler_G.step()

                writer.add_scalar(f"{fold}/train/loss_G", losses_G, epoch)
                writer.add_scalar(f"{fold}/train/loss_D", losses_D, epoch)
                writer.add_scalar(f"{fold}/train/loss_BCE", losses_BCE, epoch)
                writer.add_scalar(f"{fold}/train/loss_MSE", losses_MSE, epoch)
                writer.add_scalar(f"{fold}/val/loss_MSE", losses_MSE_val, epoch)
                writer.add_scalar(f"{fold}/val/psnr", psnr_val, epoch)

                if (epoch + 1) % save_freq == 0:
                    if epoch > 20:
                        torch.save(generator.state_dict(), pth_path + "/" + str(fold) + "epoch" + str(epoch) + ".pth")

                results = [losses_MSE_val, psnr_val]
                list_results.append(results)
                df = pd.DataFrame(
                    {
                        "MSE_val": list(np.array(list_results).T[0]),
                        "psnr_val": list(np.array(list_results).T[1]),
                    }
                )
                df.to_csv(result_path + "/" + str(fold) + ".csv")
