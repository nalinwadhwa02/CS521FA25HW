# %% [markdown]
# ## Interval Analysis
# %%
# !pip install tensorboardX

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torchsummary import summary
# from tensorboardX import SummaryWriter

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 64

np.random.seed(42)
torch.manual_seed(42)


## Dataloaders
train_dataset = datasets.MNIST('mnist_data/', train=True, download=True, transform=transforms.Compose(
    [transforms.ToTensor()]
))
test_dataset = datasets.MNIST('mnist_data/', train=False, download=True, transform=transforms.Compose(
    [transforms.ToTensor()]
))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

## Simple NN. You can change this if you want. If you change it, mention the architectural details in your report.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(28*28, 200)
        self.fc2 = nn.Linear(200,10)

    def forward(self, x):
        x = x.view((-1, 28*28))
        x = F.relu(self.fc(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=-1) # added softmax for probabilities
        return x

class Normalize(nn.Module):
    def forward(self, x):
        return (x - 0.1307)/0.3081

# Add the data normalization as a first "layer" to the network
# this allows us to search for adverserial examples to the real image, rather than
# to the normalized image
prev_model = nn.Sequential(Normalize(), Net()).to(device)

model = nn.Sequential(
    Normalize(),
    nn.Flatten(),
    nn.Linear(28*28,200),
    nn.ReLU(),
    nn.Linear(200, 10),
    nn.Softmax(dim=-1),
)

model = model.to(device)
model.train()


# %%
summary(model, (28, 28))
summary(prev_model, (28, 28))

# %%
def train_model(model, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.3f}')

def test_model(model):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Accuracy on images: {100 * correct / total}')

# %%
train_model(model, 10)
test_model(model)

# %% [markdown]
# ### Write the interval analysis for the simple model

# %%
## TODO: Write the interval analysis for the simple model
## you can use https://github.com/Zinoex/bound_propagation

### Custom Bounding framework
# I wrote a simple factory similar to one in bound_propogation to get bounds for any simple nn model

class Bounds:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

class BoundedModule(nn.Module):
    def __init__(self, model_module):
        super().__init__()
        self.model_module = model_module
    def forward(self, x):
        return self.model_module(x)
    def interval_forward(self, bounds):
        raise NotImplementedError


class BoundedLinear(BoundedModule):
    def interval_forward(self, bounds):
        weights = self.model_module.weight
        bias = self.model_module.bias

        weight_pos = torch.clamp(weights, min=0) # tensor of positive wieghts | 0
        weight_neg = torch.clamp(weights, max=0) # tensor of negative weights | 0
        
        # Interval arithmetic for y = Wx + b
        lower = F.linear(bounds.lower, weight_pos, bias) + F.linear(bounds.upper, weight_neg, None)
        upper = F.linear(bounds.upper, weight_pos, bias) + F.linear(bounds.lower, weight_neg, None)
        
        return Bounds(lower, upper)

class BoundedReLU(BoundedModule):
    def interval_forward(self, bounds):
        lower = torch.clamp(bounds.lower, min=0)
        upper = torch.clamp(bounds.upper, min=0)

        return Bounds(lower, upper)

class BoundedSoftmax(BoundedModule):
    def interval_forward(self, bounds):
        dim = self.model_module.dim
        exp_lower = torch.exp(bounds.lower)
        exp_upper = torch.exp(bounds.upper)
        sum_exp_lower = exp_lower.sum(dim=dim, keepdim=True)
        sum_exp_upper = exp_upper.sum(dim=dim, keepdim=True)
        lower = exp_lower / sum_exp_upper
        upper = exp_upper / sum_exp_lower
        return Bounds(lower, upper)

class BoundedFlatten(BoundedModule):
    def interval_forward(self, bounds):
        batch_size = bounds.lower.size(0)
        lower = bounds.lower.view(batch_size, -1)
        upper = bounds.upper.view(batch_size, -1)
        return Bounds(lower, upper)

class BoundedSequential(BoundedModule):
    def interval_forward(self, bounds):
        self.bounded_modules = [get_bounded_module(child) for child in self.model_module.children()]
        
        for bounded_module in self.bounded_modules:
            bounds = bounded_module.interval_forward(bounds)
        
        return bounds

class BoundedMonotonic(BoundedModule):
    def interval_forward(self, bounds):
        # assuming that the generic module implements some sort of monotonic function
        lower = self.model_module.forward(bounds.lower)
        upper = self.model_module.forward(bounds.upper)
        return Bounds(lower, upper)


BoundedModuleRegistery = {
    nn.Linear : BoundedLinear,
    nn.ReLU : BoundedReLU,
    nn.Sequential : BoundedSequential,
    nn.Flatten : BoundedFlatten,
    nn.Softmax : BoundedSoftmax,
}

def get_bounded_module(module: nn.Module) -> BoundedModule:
    module_type = type(module)
    if module_type not in BoundedModuleRegistery:
        # print(f"ALERT: Using BoundedMonotonic for module {module} of type {module_type}. Only ignore this alert if this module is monotonic")
        return BoundedMonotonic(module)
    return BoundedModuleRegistery[module_type](module)

def get_input_bounds(x, eps):
    lower = torch.clamp(x-eps, min=0, max=1)
    upper = torch.clamp(x+eps, min=0, max=1)
    return Bounds(lower, upper)

# %%
bounded_model = get_bounded_module(model)

# %%
def test_model_robustness(bounded_model, eps):
    bounded_model.eval()
    with torch.no_grad():
        correct = 0
        robust = 0
        total = 0
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = bounded_model.forward(images)
            perterbed_images_bounds = get_input_bounds(images, eps)
            output_bounds = bounded_model.interval_forward(perterbed_images_bounds)
            _, predicted = torch.max(outputs.data, 1)
            # check to see if any lower bound is larger than all the rest upper bounds
            idx = torch.arange(output_bounds.lower.size(0), device=device)
            lower_true = output_bounds.lower[idx, labels]
            upper_masked = output_bounds.upper.clone()
            upper_masked[idx, labels] = float('-inf')
            upper_masked_max, _ = upper_masked.max(dim=1)
            robust_labels = (lower_true > upper_masked_max).sum().item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            robust += robust_labels
        print(f"Eps: {eps}")
        print(f"Total: {total}")
        print(f'Accuracy on images: {correct} ({100 * correct / total}%)')
        print(f'Robustness accuracy on images: {robust} ({100 * robust / total}%)')

test_model_robustness(bounded_model, 0.01)

# %%
for eps in np.arange(0.01, 0.11, 0.01):
    test_model_robustness(bounded_model, eps)
    print("-"*50)

# %%



