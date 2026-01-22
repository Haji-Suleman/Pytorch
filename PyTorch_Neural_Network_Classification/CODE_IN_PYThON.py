## 1. Make classification data and get it ready
import sklearn
from sklearn.datasets import make_circles
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"


n_samples = 1000

X, y = make_circles(n_samples, noise=0.03, random_state=42)


len(X), len(y)


print(f"First five samples of X:\n{X[:5]}")
print(f"First five sample of y: {y[:5]}")


import pandas as pd

circles = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 0], "label": y})
circles.head(10)


import matplotlib.pyplot as plt

plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)


if X.dtype == "float64":
    X = torch.from_numpy(X).type(torch.float32)
    y = torch.from_numpy(y).type(dtype=torch.float32)

X[:5], y[:5]


# SPlit DATa
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


len(X_train), len(X_test), len(y_train), len(y_test)


## 1. Building a model
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # 2. Create two nn.Linear
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, X):
        return self.layer_2(self.layer_1(X))


model_0 = CircleModelV0().to(device)

model_0


model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1).to(device),
)


model_0.state_dict()


with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))
print(f"Length of prediction: {len(untrained_preds)},shape:{untrained_preds.shape} ")
print(f"Length of test samples:{len(X_test)}, Shape:{X_test.shape}")
print(f"\n First 10 prediction :\n{untrained_preds}")
print(f"\nFirst 10 labels:\n{y_test[:10]}")


lossfn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.1)


model_0.state_dict()


# Accuracy
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


# 3. Train model
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]

y_logits


y_pred_probs = torch.sigmoid(y_logits)
y_pred_probs


# Find the predicted
y_preds = torch.round(y_pred_probs)


y_preds_label = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))
print(torch.eq(y_preds.squeeze(), y_preds_label.squeeze()))


torch.cuda.manual_seed_all(42)
torch.manual_seed(42)

epochs = 100
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


## Training loop

for epoch in range(epochs):
    model_0.train()
    # 1.Forward pass
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    # 2.Calculate loss/accuracy
    loss = lossfn(y_logits, y_pred)
    acc = accuracy_fn(y_train, y_pred)
    # 3. optimize zero grad
    optimizer.zero_grad()
    # 4.
    loss.backward()
    # 5.
    optimizer.step()

    ### Testing
    model_0.eval()
    with torch.inference_mode():
        # 1. Forward Pass
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        # 2. Calculate test loss/ac
        test_loss = lossfn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_pred)
        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch} | Loss: {loss:5f}, Acc:{acc:2f}% | Test loss: {test_loss:5f}, Test acc:{test_acc:2f}%"
            )


import requests
from pathlib import Path

if Path("helper.py").is_file():
    print("Helper function.py already exists")
else:
    print("Downloading helper function")
    request = requests.get(
        "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py"
    )
    with open("helper.py", "wb") as f:
        f.write(request.content)
        f.close()
