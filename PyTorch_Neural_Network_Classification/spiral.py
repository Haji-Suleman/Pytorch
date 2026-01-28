from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split

N_SAMPLES = 1000
RANDOM_SEED = 42
X, y = make_swiss_roll(n_samples=N_SAMPLES, noise=0.1, random_state=RANDOM_SEED)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap="viridis")
plt.show()

device = "cuda" if torch.cuda.is_available() else "cpu"

X = torch.from_numpy(X).type(dtype=torch.float32)
y = torch.from_numpy(y).type(dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)


class Swiss_Roll_Model(nn.Module):
    def __init__():
        super().__init__()
        