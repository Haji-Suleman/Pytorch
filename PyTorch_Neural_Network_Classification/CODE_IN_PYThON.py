# 1️⃣ Import libraries
import torch
from torch import nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2️⃣ Create classification data
n_samples = 1000
X, y = make_circles(n_samples=n_samples, noise=0.03, random_state=42)

# Quick check
print(f"First five samples of X:\n{X[:5]}")
print(f"First five labels of y:\n{y[:5]}")

# Plot the data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()

# Convert to torch tensors
X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

# 3️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


# 4️⃣ Build the model
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(2, 5)
        self.layer_2 = nn.Linear(5, 1)

    def forward(self, X):
        return self.layer_2(self.layer_1(X))  # logits


model_0 = CircleModelV0().to(device)
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=126),
    nn.Linear(in_features=126, out_features=1).to(device),
)

# 5️⃣ Loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model_0.parameters(), lr=0.01)


# 6️⃣ Accuracy function
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


# 7️⃣ Training loop
epochs = 100

for epoch in range(epochs):
    model_0.train()
    # Forward pass
    y_logits = model_0(X_train).squeeze()
    loss = loss_fn(y_logits, y_train)

    # Predictions and accuracy
    y_pred = torch.round(torch.sigmoid(y_logits))
    acc = accuracy_fn(y_train, y_pred)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Testing
    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test).squeeze()
        test_loss = loss_fn(test_logits, y_test)
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_acc = accuracy_fn(y_test, test_pred)

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch} | "
            f"Train Loss: {loss:.4f}, Train Acc: {acc:.2f}% | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
        )

# 8️⃣ Make some predictions
with torch.inference_mode():
    sample_logits = model_0(X_test[:5]).squeeze()
    sample_probs = torch.sigmoid(sample_logits)
    sample_preds = torch.round(sample_probs)

print("\nSample Predictions:", sample_preds)
print("True Labels:", y_test[:5])

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
from helper import plot_predictions, plot_decision_boundary

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
# plt.show()


class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, X):
        # <-
        # z = self.layer_1(X)
        # z = self.layer_2(X)
        # z = self.layer_3(X)
        # ->
        return self.layer_3(self.layer_2(self.layer_1(X)))


model_1 = CircleModelV1().to(device=device)
print(model_1.state_dict())


loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)


torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 1000

X_train, y_train, X_test, y_test = (
    X_train.to(device),
    y_train.to(device),
    X_test.to(device),
    y_test.to(device),
)

for epoch in range(epochs):
    model_1.train()
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_1.eval()
    with torch.inference_mode():
        test_logits = model_1(X_train).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f} | Test loss: {test_loss:2f}"
        )
