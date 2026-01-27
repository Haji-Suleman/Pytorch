import sklearn
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split

NUM_FEATURE = 1000
RANDOM_SEED = 42
X, y = make_moons(n_samples=NUM_FEATURE, noise=0.3, random_state=RANDOM_SEED)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
