import math

import numpy as np
import torch
import torch.nn as nn

def get_samples(n_samples):
    x0 = np.random.random(3)-0.5
    x0 /= np.linalg.norm(x0)

    X0 = np.ndarray((n_samples, 3), dtype=np.float32)
    X1 = np.ndarray((n_samples, 3), dtype=np.float32)
    targets = np.ndarray((n_samples, ), dtype=np.float32)

    for i in range(n_samples):
        x1 = np.random.random(3)-0.5
        x1 /= np.linalg.norm(x0)

        X0[i] = x0
        X1[i] = x1

        d = math.acos(np.dot(x0, x1))
        targets[i] = d 
    X0 = torch.from_numpy(X0)
    X1 = torch.from_numpy(X1)
    targets = torch.from_numpy(targets)
    return X0, X1, targets

class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(3, 64)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, 1)
    
    def forward(self, x0, x1):
        x0 = torch.relu(self.layer1(x0))
        x1 = torch.relu(self.layer1(x1))

        x = torch.cat([x0, x1], dim=1)
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.layer4(x)
        return x

model = Model()
optim = torch.optim.SGD(model.parameters(), lr=0.001)

n_iters = 10000
batch_size = 64
model.train()
for i in range(n_iters):
    optim.zero_grad()
    X0, X1, targets = get_samples(batch_size)
    pred = torch.flatten(model(X0, X1))
    loss = torch.nn.functional.mse_loss(pred, targets)
    loss.backward()

    print(f'{i}: {loss.item()}')

    optim.step()
