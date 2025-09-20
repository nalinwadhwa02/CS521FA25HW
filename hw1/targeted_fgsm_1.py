import torch
import torch.nn as nn


# fix seed so that random initialization always performs the same
torch.manual_seed(13)


# create the model N as described in the question
N = nn.Sequential(nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 3, bias=False))

# random input
x = torch.rand((1,10)) # the first dimension is the batch size; the following dimensions the actual dimension of the data
x.requires_grad_() # this is required so we can compute the gradient w.r.t x

t = 1

epsReal = 1 #depending on your data this might be large or small
eps = epsReal - 1e-7 # small constant to offset floating-point erros

original_class = N(x).argmax(dim=1).item() # TO LEARN: make sure you understand this expression
print("Original Class: ", original_class)
print("Scores: ", N(x).data)
assert(original_class == 2)

# targeted Iterative FGSM
iterations = 100
alpha = 0.25

L = nn.CrossEntropyLoss()

adv_x = x.clone().detach()
adv_x.requires_grad_(True)

for iteration in range(iterations):
    adv_x.requires_grad_(True)
    N.zero_grad()
    loss = L(N(adv_x), torch.tensor([t], dtype=torch.long))
    loss.backward()
    with torch.no_grad():
        adv_x = adv_x - (alpha * adv_x.grad.sign())
        # find delta, clamp with eps
        delta = adv_x - x
        delta = torch.clamp(delta, -eps, eps)
        adv_x = x + delta

        # check if attack is successful
        new_class = N(adv_x).argmax(dim=1).item()
        if new_class == t:
            break

print(f"Iterations: {iteration+1}")
new_class = N(adv_x).argmax(dim=1).item()
print("New Class: ", new_class)
print("Adv Scores: ", N(adv_x).data)
if new_class != t:
    print("Failed to generate adversarial example")

print("Distance: ", torch.norm((x-adv_x),  p=float('inf')).data)
assert( torch.norm((x-adv_x), p=float('inf')) <= epsReal)
