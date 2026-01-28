from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_SAMPLES = 1000
RANDOM_SEED = 42

X, y = make_moons(n_samples=NUM_SAMPLES, random_state=RANDOM_SEED, noise=0.05)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return correct / len(y_true) * 100


class MoonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(2, 10)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(10, 10)
        self.layer_3 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        return self.layer_3(x)


torch.manual_seed(RANDOM_SEED)
model = MoonModel().to(device)

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 200

for epoch in range(epochs):
    model.train()

    logits = model(X_train).squeeze()
    loss = loss_fn(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
with torch.inference_mode():
    test_logits = model(X_test).squeeze()

    test_preds = torch.round(torch.sigmoid(test_logits))

    test_acc = accuracy_fn(y_test, test_preds)

print(f"Test Accuracy: {test_acc:.2f}%")

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm")
plt.show()
