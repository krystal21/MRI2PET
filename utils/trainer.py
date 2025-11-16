# -*- coding: utf-8 -*-
"""
时间：2022年03月17日
"""
import logging
import torch
import torchio as tio
from torch.cuda.amp import GradScaler
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, roc_curve
import numpy as np
import math
import torch.nn.functional as F


def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    # ctof = 1.2 * tpr + 1 - fpr
    ctof = tpr + 1 - fpr
    # ctof = np.abs(tpr+fpr-1)
    optimal_idx = np.argmax(ctof)
    # optimal_idx = np.argmin(ctof)
    optimal_threshold = threshold[optimal_idx]
    return optimal_threshold


def mse_to_psnr(mse, max_pixel_value=1):
    return 10 * math.log10((max_pixel_value**2) / mse)


def get_metric(y_true, y_pred_prob):
    auc = roc_auc_score(y_true, y_pred_prob)
    threshold = Find_Optimal_Cutoff(y_true, y_pred_prob)
    y_pred_pre = [1 if p >= threshold else 0 for p in y_pred_prob]
    recall = recall_score(y_true, y_pred_pre, pos_label=1)
    acc = accuracy_score(y_true, y_pred_pre)
    spe = recall_score(y_true, y_pred_pre, pos_label=0)
    return auc, acc, recall, spe


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_step1(dataloader, generator, discriminator, optimizer_D, optimizer_G, adversarial_loss, error_loss):
    for patches_batch in dataloader:
        inputs = patches_batch["T1"][tio.DATA]
        # targets = patches_batch["pet"][tio.DATA]
        # mask = patches_batch["mask"][tio.DATA]
        bsz = inputs.shape[0]
        print(len(dataloader), bsz)


def train_step(dataloader, generator, discriminator, optimizer_D, optimizer_G, adversarial_loss, error_loss):
    losses_G = AverageMeter()
    losses_D = AverageMeter()
    losses_BCE = AverageMeter()
    losses_MSE = AverageMeter()
    psnrs = AverageMeter()
    scaler = GradScaler()

    for patches_batch in dataloader:
        generator.train()
        # targets, inputs, mask = patches_batch
        inputs = patches_batch["T1"][tio.DATA]
        targets = patches_batch["pet"][tio.DATA]
        mask = patches_batch["mask"][tio.DATA]
        inputs = inputs.cuda()
        targets = targets.cuda()
        mask = mask.cuda()
        inputs = inputs * mask
        targets = targets * mask

        # Adversarial ground truths
        bsz = inputs.shape[0]
        # valid = torch.ones(bsz, 1, 8, 8, 8, requires_grad=False).cuda()
        # fake = torch.zeros(bsz, 1, 8, 8, 8, requires_grad=False).cuda()
        valid = torch.ones(bsz, 1, requires_grad=False).cuda()
        fake = torch.zeros(bsz, 1, requires_grad=False).cuda()

        # Train Generator
        optimizer_G.zero_grad()
        with torch.cuda.amp.autocast():
            gen_imgs = generator(inputs)
            gen_imgs = gen_imgs * mask
            bce_loss = adversarial_loss(discriminator(gen_imgs, inputs), valid)
            mse_loss = error_loss(gen_imgs, targets)
        g_loss = bce_loss + 20 * mse_loss
        losses_G.update(g_loss.item(), bsz)
        losses_BCE.update(bce_loss.item(), bsz)
        psnr = mse_to_psnr(mse_loss.item())
        psnrs.update(psnr, bsz)
        losses_MSE.update(mse_loss.item(), bsz)
        scaler.scale(g_loss).backward()
        scaler.step(optimizer_G)
        scaler.update()

        # Train Discriminator
        optimizer_D.zero_grad()
        with torch.cuda.amp.autocast():
            real_loss = adversarial_loss(discriminator(targets, inputs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), inputs), fake)
        d_loss = (real_loss + fake_loss) / 2
        losses_D.update(d_loss.item(), bsz)
        scaler.scale(d_loss).backward()
        scaler.step(optimizer_D)
        scaler.update()
        # d_loss.backward()
        # optimizer_D.step()

    logging.info(f"train_epoch: loss_g:{losses_G.avg:.4f} loss_d:{losses_D.avg:.4f}")
    return losses_G.avg, losses_D.avg, losses_BCE.avg, losses_MSE.avg, psnrs.avg


def train_step_text(dataloader, generator, discriminator, optimizer_D, optimizer_G, adversarial_loss, error_loss):
    losses_G = AverageMeter()
    losses_D = AverageMeter()
    losses_BCE = AverageMeter()
    losses_MSE = AverageMeter()
    psnrs = AverageMeter()
    scaler = GradScaler()
    for patches_batch in dataloader:
        generator.train()
        inputs = patches_batch["T1"][tio.DATA]
        targets = patches_batch["pet"][tio.DATA]
        mask = patches_batch["mask"][tio.DATA]
        bloods = patches_batch["blood"]
        bloods1 = patches_batch["blood1"]

        inputs = inputs.cuda()
        targets = targets.cuda()
        mask = mask.cuda()
        control = bloods.cuda()
        control1 = bloods1.cuda()
        inputs = inputs * mask
        targets = targets * mask

        bsz = inputs.shape[0]
        valid = torch.ones(bsz, 1, requires_grad=False).cuda()
        fake = torch.zeros(bsz, 1, requires_grad=False).cuda()

        #  Train Generator
        optimizer_G.zero_grad()
        with torch.cuda.amp.autocast():
            gen_imgs = generator(inputs, control)
            gen_imgs = gen_imgs * mask
            # Loss measures generator's ability to fool the discriminator
            bce_loss = adversarial_loss(discriminator(gen_imgs, inputs, control1), valid)
            mse_loss = error_loss(gen_imgs, targets)
        g_loss = bce_loss + 20 * mse_loss
        losses_G.update(g_loss.item(), bsz)
        losses_BCE.update(bce_loss.item(), bsz)
        losses_MSE.update(mse_loss.item(), bsz)
        psnr = mse_to_psnr(mse_loss.item())
        psnrs.update(psnr, bsz)
        # g_loss.backward()
        # optimizer_G.step()
        scaler.scale(g_loss).backward()
        scaler.step(optimizer_G)
        scaler.update()

        # Train Discriminator
        optimizer_D.zero_grad()
        # Measure discriminator's ability to classify real from generated samples
        with torch.cuda.amp.autocast():
            real_loss = adversarial_loss(discriminator(targets, inputs, control1), valid)
            # fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), inputs, control1), fake)
        d_loss = (real_loss + fake_loss) / 2
        losses_D.update(d_loss.item(), bsz)
        # d_loss.backward()
        # optimizer_D.step()
        scaler.scale(d_loss).backward()
        scaler.step(optimizer_D)
        scaler.update()

    logging.info(f"train_epoch: loss_g:{losses_G.avg:.4f} loss_d:{losses_D.avg:.4f}")
    return losses_G.avg, losses_D.avg, losses_BCE.avg, losses_MSE.avg, psnrs.avg


def val_step_text(dataloader, generator, error_loss):
    psnrs = AverageMeter()
    losses_MSE = AverageMeter()
    for patches_batch in dataloader:
        generator.eval()
        with torch.no_grad():
            # targets, inputs, mask, bloods = patches_batch
            inputs = patches_batch["T1"][tio.DATA]
            targets = patches_batch["pet"][tio.DATA]
            mask = patches_batch["mask"][tio.DATA]
            bloods = patches_batch["blood"]
            inputs = inputs.cuda()
            targets = targets.cuda()
            mask = mask.cuda()
            inputs = inputs * mask
            targets = targets * mask
            control = bloods.cuda()

            # Adversarial ground truths
            bsz = inputs.shape[0]
            real_imgs = targets
            #  val Generator
            with torch.cuda.amp.autocast():
                gen_imgs = generator(inputs, control)
                gen_imgs = gen_imgs * mask
                mse_loss = error_loss(gen_imgs, real_imgs)
            psnr = mse_to_psnr(mse_loss.item())
            psnrs.update(psnr, bsz)
            losses_MSE.update(mse_loss.item(), bsz)
    logging.info(f"val_epoch: loss_mse:{losses_MSE.avg:.4f}")
    return losses_MSE.avg, psnrs.avg


def val_step(dataloader, generator, error_loss):
    losses_MSE = AverageMeter()
    psnrs = AverageMeter()

    for patches_batch in dataloader:
        generator.eval()
        with torch.no_grad():
            inputs = patches_batch["T1"][tio.DATA]
            targets = patches_batch["pet"][tio.DATA]
            mask = patches_batch["mask"][tio.DATA]
            inputs = inputs.cuda()
            targets = targets.cuda()
            mask = mask.cuda()
            inputs = inputs * mask
            targets = targets * mask

            # Adversarial ground truths
            bsz = inputs.shape[0]
            real_imgs = targets
            #  val Generator
            with torch.cuda.amp.autocast():
                gen_imgs = generator(inputs)
                gen_imgs = gen_imgs * mask
                mse_loss = error_loss(gen_imgs, real_imgs)
            psnr = mse_to_psnr(mse_loss.item(), 1)
            psnrs.update(psnr, bsz)
            losses_MSE.update(mse_loss.item(), bsz)

    logging.info(f"val_epoch: loss_mse:{losses_MSE.avg:.4f} psnr:{psnrs.avg:.4f} ")
    return losses_MSE.avg, psnrs.avg


def train_clip(dataloader, model, optimizer):
    losses = AverageMeter()
    model.train()
    for patches_batch in dataloader:
        image = patches_batch["pet"][tio.DATA]
        mask = patches_batch["mask"][tio.DATA]
        bloods = patches_batch["blood"]
        image = image.cuda()
        mask = mask.cuda()
        text_embedding = bloods.cuda()
        image = image * mask

        loss = model(image, text_embedding)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bsz = image.shape[0]
        losses.update(loss.item(), bsz)
    logging.info(f"train_epoch: {losses.avg:.4f}")
    return losses.avg


def val_clip(dataloader, model):
    losses = AverageMeter()
    model.eval()
    for patches_batch in dataloader:
        image = patches_batch["pet"][tio.DATA]
        mask = patches_batch["mask"][tio.DATA]
        bloods = patches_batch["blood"]
        image = image.cuda()
        mask = mask.cuda()
        text_embedding = bloods.cuda()
        image = image * mask
        with torch.no_grad():
            loss = model(image, text_embedding)
        bsz = image.shape[0]
        losses.update(loss.item(), bsz)
    logging.info(f"val_epoch: {losses.avg:.4f}")
    return losses.avg


def train_vae(dataloader, model, optimizer, criterion):
    losses = AverageMeter()
    model.train()
    for patches_batch in dataloader:
        image = patches_batch["pet"][tio.DATA]
        mask = patches_batch["mask"][tio.DATA]
        image = image.cuda()
        mask = mask.cuda()
        image = image * mask

        # 前向传播
        outputs = model(image)
        # 计算损失
        loss = criterion(outputs, image)  # 计算重构损失

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bsz = image.shape[0]
        losses.update(loss.item(), bsz)
    logging.info(f"train_epoch: {losses.avg:.4f}")
    return losses.avg


def val_vae(dataloader, model, criterion):
    losses = AverageMeter()
    model.eval()
    for patches_batch in dataloader:
        image = patches_batch["pet"][tio.DATA]
        mask = patches_batch["mask"][tio.DATA]
        image = image.cuda()
        mask = mask.cuda()
        image = image * mask
        with torch.no_grad():
            # 前向传播
            outputs = model(image)
            # 计算损失
            loss = criterion(outputs, image)  # 计算重构损失

        bsz = image.shape[0]
        losses.update(loss.item(), bsz)
    logging.info(f"val_epoch: {losses.avg:.4f}")
    return losses.avg


def train_cls(dataloader, model, optimizer, criterion):
    losses = AverageMeter()
    # scaler = GradScaler()
    model.train()
    y_true = []
    y_pred_prob = []
    for patches_batch in dataloader:
        image = patches_batch["pet"][tio.DATA]

        # mask = patches_batch["mask"][tio.DATA]
        labels = patches_batch["label"]

        image = image.cuda()
        # mask = mask.cuda()
        labels = labels.cuda()
        # image = image * mask
        # 前向传播
        optimizer.zero_grad()
        # with torch.cuda.amp.autocast():
        outputs = model(image)
        loss = criterion(outputs, labels)
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bsz = image.shape[0]
        _, predicted = outputs.max(1)
        losses.update(loss.item(), bsz)

        y_true.extend(labels.cpu().tolist())
        y_pred_prob.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().tolist())
    auc, acc, recall, spe = get_metric(y_true, y_pred_prob)
    logging.info(f"train_epoch: {losses.avg:.4f}, acc:{acc:.4f}, auc:{auc:.4f}, spe:{spe:.4f}, recall:{recall:.4f}")
    return losses.avg, auc, acc, recall, spe


def val_cls(dataloader, model, criterion):
    losses = AverageMeter()
    y_true = []
    y_pred_prob = []
    model.eval()
    for patches_batch in dataloader:
        image = patches_batch["pet"][tio.DATA]
        # mask = patches_batch["mask"][tio.DATA]
        labels = patches_batch["label"]

        image = image.cuda()
        # mask = mask.cuda()
        labels = labels.cuda()
        # image = image * mask
        with torch.no_grad():
            # with torch.cuda.amp.autocast():
            outputs = model(image)
            loss = criterion(outputs, labels)

        bsz = image.shape[0]
        _, predicted = outputs.max(1)
        losses.update(loss.item(), bsz)
        y_true.extend(labels.cpu().tolist())
        y_pred_prob.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().tolist())
    auc, acc, recall, spe = get_metric(y_true, y_pred_prob)
    logging.info(f"val_epoch: {losses.avg:.4f}, acc:{acc:.4f}, auc:{auc:.4f}, spe:{spe:.4f}, recall:{recall:.4f}")
    return losses.avg, auc, acc, recall, spe


def train_mcf(dataloader, model, optimizer, criterion):
    losses = AverageMeter()
    # scaler = GradScaler()
    model.train()
    y_true = []
    y_pred_prob = []
    for patches_batch in dataloader:
        image = patches_batch["pet"][tio.DATA]
        bloods = patches_batch["blood"]
        labels = patches_batch["label"]

        image = image.cuda()
        bloods = bloods.cuda()
        labels = labels.cuda()
        # 前向传播
        optimizer.zero_grad()
        # with torch.cuda.amp.autocast():
        outputs1, outputs2 = model(image, bloods)
        # loss1 = criterion(outputs1, labels)
        # loss2 = criterion(outputs2, labels)

        # outputs11 = F.softmax(outputs1, dim=1)
        # outputs12 = F.softmax(outputs2, dim=1)
        # outputs = (outputs11 + outputs12) / 2
        # outputsx = torch.log(F.softmax(outputs, dim=1))
        outputs = (outputs1 + outputs2) / 2
        loss3 = criterion(outputs, labels)
        loss = loss3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bsz = image.shape[0]
        losses.update(loss.item(), bsz)

        y_true.extend(labels.cpu().tolist())
        y_pred_prob.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().tolist())
        # y_pred_prob.extend(outputs[:, 1].cpu().tolist())
    auc, acc, recall, spe = get_metric(y_true, y_pred_prob)
    logging.info(f"train_epoch: {losses.avg:.4f}, acc:{acc:.4f}, auc:{auc:.4f}, spe:{spe:.4f}, recall:{recall:.4f}")
    return losses.avg, auc, acc, recall, spe


def val_mcf(dataloader, model, criterion):
    losses = AverageMeter()
    y_true = []
    y_pred_prob = []
    model.eval()
    for patches_batch in dataloader:
        image = patches_batch["pet"][tio.DATA]
        bloods = patches_batch["blood"]
        labels = patches_batch["label"]

        image = image.cuda()
        bloods = bloods.cuda()
        labels = labels.cuda()
        # image = image * mask
        with torch.no_grad():
            # with torch.cuda.amp.autocast():
            outputs1, outputs2 = model(image, bloods)
            # outputs11 = F.softmax(outputs1, dim=1)
            # outputs12 = F.softmax(outputs2, dim=1)
            # outputs = (outputs11 + outputs12) / 2
            outputs = (outputs1 + outputs2) / 2
            loss = criterion(outputs, labels)

        bsz = image.shape[0]
        losses.update(loss.item(), bsz)
        y_true.extend(labels.cpu().tolist())
        y_pred_prob.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().tolist())
        # y_pred_prob.extend(outputs[:, 1].cpu().tolist())
    auc, acc, recall, spe = get_metric(y_true, y_pred_prob)
    logging.info(f"val_epoch: {losses.avg:.4f}, acc:{acc:.4f}, auc:{auc:.4f}, spe:{spe:.4f}, recall:{recall:.4f}")
    return losses.avg, auc, acc, recall, spe
