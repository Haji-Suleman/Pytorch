import sklearn
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FEATURE = 1000
RANDOM_SEED = 42
X, y = make_moons(n_samples=NUM_FEATURE, noise=0.3, random_state=RANDOM_SEED)

X = torch.from_numpy(X).type(dtype=torch.float32)
y = torch.from_numpy(X).type(dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y)


plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()


class MoonModel(nn.Mode):
    def __init__(self):
        super.__init__()
        self.layer_1 = (nn.Linear(in_features=2, out_features=10),)
        self.layer_2 = (nn.Linear(in_features=10, out_features=10),)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, X: torch.tensor) -> torch.tensor:
        return self.layer_3(self.layer_2(self.layer_1(X)))


torch.manual_seed(RANDOM_SEED)

