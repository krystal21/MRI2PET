import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from torch.optim import lr_scheduler


class CustomDataset(Dataset):
    def __init__(self, X_data, Y_data, idx):
        self.X_data = X_data
        self.Y_data = Y_data
        self.idx = idx

    def __getitem__(self, index):
        sample = {"X": self.X_data[index], "Y": self.Y_data[index], "idx": self.idx[index]}
        return sample

    def __len__(self):
        return len(self.X_data)


def train_model(model, train_loader, criterion, optimizer):
    model.train()
    predictions = []
    true_labels = []
    idxs = []
    running_loss = 0.0
    for batch in train_loader:
        labels = batch["Y"].cuda()
        inputs = batch["X"].cuda()
        idx = batch["idx"]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        predictions.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())
        # predictions.extend(predicted_probs.detach().cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        idxs.append(idx)
    return true_labels, predictions, running_loss / len(train_loader), idxs


def validate_model(model, val_loader, criterion):
    model.eval()
    predictions = []
    true_labels = []
    idxs = []
    running_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            labels = batch["Y"].cuda()
            inputs = batch["X"].cuda()
            idx = batch["idx"]
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            running_loss += loss.item()
            predictions.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            idxs.append(idx[0])

    return true_labels, predictions, running_loss / len(val_loader), idxs


# 定义ANN模型类
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


ss = []
df_m = pd.DataFrame()
for fold in range(1):
    fold = fold
    # print("--------------------------------------------------")
    # 读取CSV文件
    df_train = pd.read_csv(f"data/HS/fold_test/{fold}train.csv")
    df_val = pd.read_csv(f"data/HS/fold_test/{fold}val.csv")
    df_train = df_train[df_train["labels1"] != 2]
    df_val = df_val[df_val["labels1"] != 2]

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
    idx_train = df_train["ID"].tolist()
    idx_val = df_val["ID"].tolist()

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_dataset = CustomDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(Y_train.values, dtype=torch.float32), idx_train
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = CustomDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(Y_val.values, dtype=torch.float32), idx_val
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 初始化ANN模型
    model = ANN(len(blood_str), 2).cuda()

    # 定义损失函数和优化器
    n1 = df_val["labels"].sum()
    n0 = len(df_val) - df_val["labels"].sum()
    ratio = 2.0
    weight = torch.tensor([1.0, float(ratio)])
    criterion = nn.CrossEntropyLoss(weight=weight).cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.03)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00001)

    # 训练和验证模型
    num_epochs = 100
    best_rp = 0
    best_auc = 0
    acc_l = []
    auc_l = []
    recall_l = []
    spe_l = []

    for epoch in range(num_epochs):
        true_labels1, predictions1, train_loss, idxs = train_model(model, train_loader, criterion, optimizer)
        true_labels, predictions, val_loss, idxs_val = validate_model(model, val_loader, criterion)
        auc_val = roc_auc_score(true_labels, predictions)
        acc_val = accuracy_score(true_labels, np.round(predictions))
        recall_val = recall_score(true_labels, np.round(predictions))
        spe_val = recall_score(true_labels, np.round(predictions), pos_label=0)
        acc_train = accuracy_score(true_labels1, np.round(predictions1))
        scheduler.step()
        # print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss}, acc: {acc_train},")
        # print(f"AUC: {auc_val}, Val acc: {acc_val}, recall: {recall_val},spe:{spe_val}")

        acc_l.append(acc_val)
        auc_l.append(auc_val)
        recall_l.append(recall_val)
        spe_l.append(spe_val)
        # test
        # rp = 1.5 * recall_val + spe_val + auc_val
        # rp = recall_val + auc_val + spe_val
        rp = auc_val
        if rp >= best_rp:
            best_rp = rp
            # print(f"AUC: {auc_val}, Val acc: {acc_val}, recall: {recall_val},spe:{spe_val}")
            predictions = [i for i in predictions]
            dfx = pd.DataFrame({"labels": true_labels, "ID": idxs_val, "pre": predictions})
            dfx.to_csv(f"exp/blood/test/{fold}.csv", index=False)
            torch.save(model.state_dict(), f"exp/blood/test/{fold}best.pth")

    #     if auc_val >= best_auc:
    #         best_auc = auc_val
    #         print(f"AUC: {auc_val}, Val acc: {acc_val}, recall: {recall_val},spe:{spe_val}")
    # print(fold, "-----------------------------------------------------------------------------")
    df = pd.DataFrame({"acc": acc_l, "auc": auc_l, "recall": recall_l, "spe": spe_l})
    # print(np.max(acc_l), np.max(auc_l))
    idx = df["acc"].idxmax()
    print(idx, df["acc"].max(), df["auc"].loc[idx], df["recall"].loc[idx], df["spe"].loc[idx])
    ss.append(df["acc"].max())


print(np.mean(ss))
