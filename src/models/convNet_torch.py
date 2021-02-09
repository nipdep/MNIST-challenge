# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd 

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
from torchvision import  transforms

# %%

class MNIST(Dataset):
    def __init__(self, src_path, transform=None):
        self.transform = transform
        df = pd.read_csv(src_path)
        y_np = df.iloc[:, 0].values
        X_np = df.iloc[:, 1:].values/255.0

        self.y = torch.from_numpy(y_np)
        X = torch.from_numpy(X_np)
        self.X = torch.reshape(X,(-1, 1, 28 ,28)).float()
    
    def __len__(self):
        #assert len(self.y) == self.X[0]
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X_sub = self.X[idx, :]
        y_sub = self.y[idx]

        if self.transform:
            X_sub = self.transform(X_sub)

        return X_sub, y_sub

# %%

transformer = transforms.Compose([
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

dataset = MNIST('../../data/datasets/train.csv')
loader = DataLoader(dataset,batch_size=20, shuffle=True, num_workers=0)

# %%
class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Linear(in_features=32*5*5, out_features=128)
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=10)
    
    def forward(self, x):
        in_size = x.size(0)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32*5*5)
        x = F.relu(self.flatten(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) ## this is the output layer.
        x = F.softmax(x)        ## this is the logits layer.
        return x 

model = ConvNet()
model

# %%

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# %%

epochs = 10
for epoch in range(epochs):

    for i, data in enumerate(loader):
        inputs, labels = data 

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f'[{epoch+1} : {i+1}] loss : {loss.item()}')
print('Finished Training')

# %%
