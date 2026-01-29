from sklearn.datasets import make_classification
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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


class Classifcation_model(nn.Module):
    def __init__(self, n_feature):
        super().__init__()
        model = nn.Sequential(
            nn.Linear(in_features=n_feature, out_features=128),
            nn.Linear(in_features=128, out_features=128),
            nn.Linear(in_features=128, out_features=4),
        )
    
