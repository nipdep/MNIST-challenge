# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import math
import numpy as np
import pandas as pd 

import torch
import torch.nn as nn
import torch.functional as F


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# %%

def convex_function(row):
    x,y = row
    z = 2*math.log(x**2+y**2+0.2) + 2
    return z

data_points = np.random.rand(300,2)*20 - 10
z = np.apply_along_axis(convex_function, 1, data_points)
dataset = torch.from_numpy(np.insert(data_points,0, z, axis=1)).float()

# %%

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
xs = dataset[:, 1]
ys = dataset[:, 2]
zs = dataset[:, 0]
ax.scatter(xs, ys, zs, marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

# %%

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.input = torch.nn.Linear(in_features=2, out_features=8)
        self.fc1 = torch.nn.Linear(8, 4)
        self.out = torch.nn.Linear(4,1)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.input(x))
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.out(x))
        return x

model = SimpleNN()

# %%

criterion = torch.nn.MSELoss()
#criterion = nn.NLLLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# %%

epochs = 5
for epoch in range(epochs):
    
        for row in dataset:
            inputs, labels = row[1:], row[0]
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

print('Finished Training')

# %%

model.state_dict()

# %%
