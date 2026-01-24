import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42


X_blob, y_blob = make_blobs(
    n_samples=1000,
    n_features=NUM_FEATURES,
    centers=NUM_CLASSES,
    cente_std=1.5,
    random_state=RANDOM_SEED,
)


X_blob = torch.from_numpy(X_blob).type(dtype=torch.float32)
y_blob = torch.from_numpy(X_blob).type(dtype=torch.float32)


X_blob_train, y_blob_train, X_blob_test, y_blob_test = train_test_split(
    X_blob, y_blob, test_size=0.2, random_state=RANDOM_SEED
)


plt.figure(figsize=(10,7))
plt.plot(X_blob[:,0], X_blob[:,1], c=y_blob,class Parrot:

 def fly(self):
   print('Parrot can fly')

 def swim(self):
   print('Parrot can not swim')

class Penguin:

 def fly(self):
   print('Penguin can not fly')

 def swim(self):
   print('Penguin can swim')

# common interface
def flying_test(bird):
  bird.fly()

#instantiate objects
blu = Parrot()
peggy = Penguin()

# passing the object
flying_test(blu)
flying_test(peggy))
