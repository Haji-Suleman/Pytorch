import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch import nn

# -------------------------------
# DATASET
# -------------------------------
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

X_blob, y_blob = make_blobs(
    n_samples=1000,
    n_features=NUM_FEATURES,
    centers=NUM_CLASSES,
    cluster_std=1.5,
    random_state=RANDOM_SEED,
)

X_blob = torch.from_numpy(X_blob).float()
y_blob = torch.from_numpy(y_blob).long()

X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(
    X_blob, y_blob, test_size=0.2, random_state=RANDOM_SEED
)

# -------------------------------
# VISUALIZE DATA
# -------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
plt.title("Blob Dataset")
plt.show()

# -------------------------------
# DEVICE
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

X_blob_train = X_blob_train.to(device)
y_blob_train = y_blob_train.to(device)
X_blob_test = X_blob_test.to(device)
y_blob_test = y_blob_test.to(device)


# -------------------------------
# ACCURACY FUNCTION
# -------------------------------
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100


# -------------------------------
# MODEL
# -------------------------------
class BlobModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, x):
        return self.net(x)


model_4 = BlobModel().to(device)

# -------------------------------
# LOSS & OPTIMIZER
# -------------------------------


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_4.parameters(), lr=0.01)

# -------------------------------
# TRAINING LOOP
# -------------------------------
epochs = 200

for epoch in range(epochs):
    model_4.train()
    y_logits = model_4(X_blob_train)
    loss = loss_fn(y_logits, y_blob_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(loss)
# -------------------------------
# TESTING
# -------------------------------
model_4.eval()
with torch.inference_mode():
    test_logits = model_4(X_blob_test)
    test_preds = torch.argmax(test_logits, dim=1)
    test_acc = accuracy_fn(y_blob_test, test_preds)

print(f"\nTest Accuracy: {test_acc:.2f}%")


# -------------------------------
# DECISION BOUNDARY
# -------------------------------
def plot_decision_boundary(model_4, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = torch.meshgrid(
        torch.linspace(x_min, x_max, 200),
        torch.linspace(y_min, y_max, 200),
        indexing="ij",
    )

    grid = torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=1).to(device)

    model_4.eval()
    with torch.inference_mode():
        preds = torch.argmax(model_4(grid), dim=1)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx.cpu(), yy.cpu(), preds.reshape(xx.shape).cpu(), alpha=0.4)
    plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=y.cpu(), s=20)
    plt.title("Decision Boundary")
    plt.show()


plot_decision_boundary(model_4, X_blob_train, y_blob_train)


