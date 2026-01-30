# -------------------------------
# Multiclass Non-linear Classification
# Author: Haji Suleman
# -------------------------------

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchmetrics
from torchmetrics import Accuracy

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

N_SAMPLES = 2000
N_FEATURES = 10
N_INFORMATIVE = 6
N_REDUNDANT = 2
N_CLASSES = 5
CLASS_SEP = 0.8
FLIP_Y = 0.05

X, y = make_classification(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    n_informative=N_INFORMATIVE,
    n_redundant=N_REDUNDANT,
    n_classes=N_CLASSES,
    class_sep=CLASS_SEP,
    flip_y=FLIP_Y,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)
scaler = StandardScaler()
X_train = torch.from_numpy(scaler.fit_transform(X_train)).float()
X_test = torch.from_numpy(scaler.transform(X_test)).float()

y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return correct / len(y_true) * 100


class ClassificationModel(nn.Module):
    def __init__(
        self, input_size, hidden1, hidden2, hidden3, output_size, dropout_p=0.2
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.bn3 = nn.BatchNorm1d(hidden3)
        self.fc4 = nn.Linear(hidden3, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, X):
        X = self.fc1(X)
        X = self.bn1(X)
        X = self.relu(X)
        X = self.dropout(X)

        X = self.fc2(X)
        X = self.bn2(X)
        X = self.relu(X)
        X = self.dropout(X)

        X = self.fc3(X)
        X = self.bn3(X)
        X = self.relu(X)
        X = self.dropout(X)

        X = self.fc4(X)
        return X


HIDDEN1 = 256
HIDDEN2 = 128
HIDDEN3 = 64
OUTPUT_SIZE = N_CLASSES

model_6 = ClassificationModel(
    input_size=N_FEATURES,
    hidden1=HIDDEN1,
    hidden2=HIDDEN2,
    hidden3=HIDDEN3,
    output_size=OUTPUT_SIZE,
    dropout_p=0.3,
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_6.parameters(), lr=0.01, weight_decay=1e-4)

EPOCHS = 200
train_losses = []

for epoch in range(EPOCHS):
    model_6.train()
    optimizer.zero_grad()
    logits = model_6(X_train)
    y_preds = torch.argmax(logits, dim=1)
    loss = loss_fn(logits, y_train)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Train Loss: {loss.item():.4f}")

model_6.eval()
with torch.inference_mode():
    test_logits = model_6(X_test)
    test_preds = torch.argmax(test_logits, dim=1)
    test_acc = accuracy_fn(y_test, test_preds)
    print(f"\nTest Accuracy: {test_acc:.2f}%")

    for cls in range(N_CLASSES):
        idx = y_test == cls
        cls_acc = accuracy_fn(y_test[idx], test_preds[idx])
        print(f"Class {cls} Accuracy: {cls_acc:.2f}%")
torchmetrics_accuracy = torchmetrics.Accuracy()
torchmetrics_accuracy(y_preds, y_test)
