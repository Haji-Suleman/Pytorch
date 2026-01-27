import sklearn
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FEATURE = 1000
RANDOM_SEED = 42
X, y = make_moons(n_samples=NUM_FEATURE, noise=0.3, random_state=RANDOM_SEED)

X = torch.from_numpy(X).type(dtype=torch.float32)
y = torch.from_numpy(y).type(dtype=torch.float32)

X_moon_train, X_moon_test, y_moon_train, y_moon_test = train_test_split(X, y)


plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()


def accuracy_fn(y_true, y_preds) -> torch.tensor:
    correct = torch.eq(y_true, y_preds).sum().item()
    return correct / len(y_true) * 100


class MoonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.ReLU = nn.ReLU()
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.ReLU = nn.ReLU()

        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, X: torch.tensor) -> torch.tensor:
        return self.layer_3(self.ReLU(self.layer_2(self.ReLU(self.layer_1(X)))))


torch.manual_seed(RANDOM_SEED)
model_5 = MoonModel()

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(params=model_5.parameters(), lr=0.1)

epochs = 200

for epoch in range(epochs):
    model_5.train()
    y_logits = model_5(X_moon_train).squeeze()
    y_preds = torch.round(torch.sigmoid(y_logits))
    loss = loss_fn(y_logits, y_moon_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model_5.eval()
with torch.inference_mode():
    test_logits = model_5(X_moon_test)
    test_preds = torch.argmax(test_logits)
    test_acc = accuracy_fn(y_true=y_moon_test, y_preds=test_logits)
plt.scatter(y_moon_test, test_logits, c=y_preds)

plt.show()
print(f"Test Accuracy {test_acc:.2f}")
