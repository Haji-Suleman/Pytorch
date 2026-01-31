import torch
from torch import nn
import matplotlib.pyplot as plt

weight = 0.3
bias = 0.9

X = torch.arange(1, 101, dtype=torch.float32).unsqueeze(1)
X = X / 100
y = (X * weight) + bias

train_size = int(0.8 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]


def plot_data(y_preds=None):
    plt.figure(figsize=(10, 8))
    plt.scatter(X_train, y_train, s=5, label="Train")
    plt.scatter(X_test, y_test, s=5, label="Test")
    if y_preds is not None:
        plt.scatter(X_test, y_preds, s=5, label="Preds")
    plt.legend()
    plt.show()


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.weight * x + self.bias


torch.manual_seed(42)
model = LinearRegressionModel()

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return correct / len(y_true) * 100


epochs = 400
for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 40 == 0:
        model.eval()
        with torch.inference_mode():
            test_pred = model(X_test)
            test_loss = loss_fn(test_pred, y_test)
        print(
            f"Epoch {epoch} | Train loss {loss.item():.4f} | Test loss {test_loss.item():.4f}"
        )
print(f"Accuracy of Function: {accuracy_fn(y_test,test_pred)}")
plot_data(y_preds=test_pred)
plt.show()
