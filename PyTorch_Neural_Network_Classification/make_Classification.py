# -------------------------------
# Multiclass Non-linear Classification
# Improved Version with Comments
# Author: Haji Suleman
# -------------------------------

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ Set random seeds for reproducibility
# -------------------------------
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

# -------------------------------
# 2️⃣ Dataset creation
# -------------------------------
# Mistake in your original code:
# - Spelling errors in variable names: N_INFROMATIVE, N_REDUNDUNT
# - Did not scale features (X_train, X_test)
# - Data may be noisy / overlapping, making training harder
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

# -------------------------------
# 3️⃣ Train/test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

# -------------------------------
# 4️⃣ Feature scaling (Important!)
# Mistake in original: You did not scale X, which slows down training and hurts accuracy
# -------------------------------
scaler = StandardScaler()
X_train = torch.from_numpy(scaler.fit_transform(X_train)).float()
X_test = torch.from_numpy(scaler.transform(X_test)).float()

# Convert labels to torch tensors
y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()


# -------------------------------
# 5️⃣ Accuracy function
# Your function is correct
# -------------------------------
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return correct / len(y_true) * 100


# -------------------------------
# 6️⃣ Improved Model
# Mistakes in original:
# - Forward pass was one long line → hard to debug
# - No Dropout → can overfit / slow learning
# - No BatchNorm → sometimes helps stability
# -------------------------------
class ClassificationModel(nn.Module):
    def __init__(
        self, input_size, hidden1, hidden2, hidden3, output_size, dropout_p=0.2
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)  # BatchNorm improves learning
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.bn3 = nn.BatchNorm1d(hidden3)
        self.fc4 = nn.Linear(hidden3, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)  # Dropout prevents overfitting

    def forward(self, X):
        # Step-by-step forward for clarity
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

        X = self.fc4(X)  # Output layer, raw logits
        return X


# -------------------------------
# 7️⃣ Instantiate model
# Original hidden sizes were okay, but increasing slightly improves learning
# -------------------------------
HIDDEN1 = 256
HIDDEN2 = 128
HIDDEN3 = 64
OUTPUT_SIZE = N_CLASSES

model = ClassificationModel(
    input_size=N_FEATURES,
    hidden1=HIDDEN1,
    hidden2=HIDDEN2,
    hidden3=HIDDEN3,
    output_size=OUTPUT_SIZE,
    dropout_p=0.3,
)

# -------------------------------
# 8️⃣ Loss & Optimizer
# Mistakes in original:
# - Learning rate too high (0.05)
# -------------------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

# -------------------------------
# 9️⃣ Training Loop
# Original code: no printing / tracking loss
# -------------------------------
EPOCHS = 300
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train)
    loss = loss_fn(logits, y_train)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Train Loss: {loss.item():.4f}")

# -------------------------------
# 10️⃣ Evaluation
# Mistakes in original:
# - Only overall accuracy
# - Did not check per-class accuracy
# -------------------------------
model.eval()
with torch.inference_mode():
    test_logits = model(X_test)
    test_preds = torch.argmax(test_logits, dim=1)
    test_acc = accuracy_fn(y_test, test_preds)
    print(f"\nTest Accuracy: {test_acc:.2f}%")

    # Per-class accuracy (important)
    for cls in range(N_CLASSES):
        idx = y_test == cls
        cls_acc = accuracy_fn(y_test[idx], test_preds[idx])
        print(f"Class {cls} Accuracy: {cls_acc:.2f}%")

# --
