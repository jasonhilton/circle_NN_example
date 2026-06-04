import sklearn
import pandas as pd
import numpy as np
from sklearn import datasets

from sklearn.datasets import make_circles, make_moons

X, Y = make_circles(n_samples=1000, noise=0.2, factor=0.3, random_state=0)
circle_df = pd.DataFrame({"X1":X[:,0],"X2":X[:,1], "Y":Y})
circle_df["Class"] =["Class-1" if y==0 else "Class-2" for y in circle_df["Y"]]

n_obs = circle_df.shape[0]

indexes = np.arange(n_obs)
np.random.shuffle(indexes)

n_train = int(n_obs*0.8)


train_ind = indexes[:n_train]
test_ind = indexes[n_train:]

circle_train = circle_df.loc[train_ind,:]
circle_test = circle_df.loc[test_ind,:]