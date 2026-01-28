from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from torch import nn

# Device (you defined it before but didn't use it)
device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_SAMPLES = 1000
RANDOM_SEED = 42

# Dataset
X, y = make_moons(n_samples=NUM_SAMPLES, random_state=RANDOM_SEED, noise=0.3)

# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

# Move data to device (IMPORTANT FIX)
X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)


# Accuracy function (this was fine)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return correct / len(y_true) * 100


class MoonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(2, 10)
        self.relu = nn.ReLU()  # FIX: don’t redefine ReLU twice
        self.layer_2 = nn.Linear(10, 10)
        self.layer_3 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        return self.layer_3(x)


torch.manual_seed(RANDOM_SEED)
model = MoonModel().to(device)  # FIX: model moved to device

loss_fn = nn.BCEWithLogitsLoss()

# FIX: learning rate was too high (0.1 breaks learning)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 200

# Training loop (this part was mostly correct)
for epoch in range(epochs):
    model.train()

    logits = model(X_train).squeeze()
    loss = loss_fn(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Evaluation (THIS was the main problem)
model.eval()
with torch.inference_mode():
    test_logits = model(X_test).squeeze()

    # FIX: argmax is WRONG for binary classification
    test_preds = torch.round(torch.sigmoid(test_logits))

    # FIX: accuracy must use predictions, not logits
    test_acc = accuracy_fn(y_test, test_preds)

print(f"Test Accuracy: {test_acc:.2f}%")

# Plot data (visual check)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm")
plt.show()
