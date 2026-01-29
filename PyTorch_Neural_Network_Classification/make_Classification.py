from sklearn.datasets import make_classification
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
N_SAMPLES = 2000
N_FEATURE = 10
N_INFROMATIVE = 6
N_REDUNDUNT = 2
N_CLASSES = 5
CLASS_SEP = 0.8
FLIP_Y = 0.05

X, y = make_classification(
    n_samples=N_SAMPLES,
    n_features=N_FEATURE,
    n_informative=N_INFROMATIVE,
    n_redundant=N_REDUNDUNT,
    n_classes=N_CLASSES,
    class_sep=CLASS_SEP,
    flip_y=FLIP_Y,
)
X = torch.from_numpy(X).type(dtype=torch.float32)
y = torch.from_numpy(y).type(dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return correct / len(y_true) * 100


class Classifcation_model(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        model = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=128),
            nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=4),
        )

    def forward(self, X):
        return self.model(X)


torch.manual_seed(RANDOM_SEED)
model_6 = Classifcation_model(n_features=N_FEATURE)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_6.parameters(), lr=0.05)

epochs = 200


for epoch in range(epochs):
    model_6.train()
    y_logits = model_6(X_train)
    loss = loss_fn(y_logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()