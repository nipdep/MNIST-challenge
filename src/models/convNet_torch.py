# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %% [markdown]

## MNIST challenge - ConvNet model : pytorch

# %%
from tqdm.notebook import tqdm
import pandas as pd 
from time import sleep

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
from torchvision import  transforms

# %%
# activate GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% [markdown]
## load dataset in mini-batch format by DataLoader

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
        assert len(self.y) == self.X.shape[0]
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
batch_size = 20
loader = DataLoader(dataset,batch_size=batch_size, shuffle=True, num_workers=0)


# %% [markdown]
## build ConvNet architecture
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
        #x = F.relu(self.fc2(x))     ## this is the output layer.
        #x = F.log_softmax(x)        ## this is the logits layer.
        x = self.fc2(x)
        return x 

model = ConvNet()
model = model.to(device)

# %% [markdown]

## define loss function, matrics and optimization algorithm.

# %%

criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()

optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# %% [markdown]

# train the model

# %%

epochs = 1
for epoch in range(epochs):
        for i, data in enumerate(loader):
            inputs, labels = data 
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'[{epoch+1} : {i+1}] loss : {loss.item()}')
print('Finished Training')

# %%

epochs = 1
for epoch in range(epochs):
    with tqdm(loader, unit='batch', position=0, leave=True) as progress_bar:
        for data in progress_bar:
            progress_bar.set_description(f'Epoch {epoch}:')

            inputs, labels = data 
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            correct = (torch.argmax(outputs, axis=1) == labels).sum().item()
            accuracy = correct / batch_size
            progress_bar.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
            #progress_bar.update()

print('Finished Training')

# %% [markdown]
## save model results

# %%

torch.save(model.state_dict(), '../../data/models/convNet_torch.pt')
# %%
torch.save(loader, '../../data/models/dataloader.pth')
# %%
