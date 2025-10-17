import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms
# from tensorboardX import SummaryWriter

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
batch_size = 64
learning_rate = 0.1
num_epochs = 5

np.random.seed(42)
torch.manual_seed(42)
# !mkdir models
# for file in ["https://drive.google.com/file/d/1bJ0C2OvVdBjyyL-KlaCPdqchFB_mvdA7/view?usp=drive_link", "https://drive.google.com/file/d/1yKdvBYFgK6AzHdh66Nuem5TRrg0Uy16a/view?usp=drive_link", "https://drive.google.com/file/d/17ISM62qq2ohpgBBm5f4DaGioIKQSozFJ/view?usp=drive_link"]:
#   !gdown --fuzzy $file
# !mv pretr* models/.


def tp_relu(x, delta=1.0):
    ind1 = (x < -1.0 * delta).float()
    ind2 = (x > delta).float()
    return 0.5 * (x + delta) * (1 - ind1) * (1 - ind2) + x * ind2


def tp_smoothed_relu(x, delta=1.0):
    ind1 = (x < -1.0 * delta).float()
    ind2 = (x > delta).float()
    return (x + delta) ** 2 / (4 * delta) * (1 - ind1) * (1 - ind2) + x * ind2


class Normalize(nn.Module):
    def __init__(self, mu, std):
        super(Normalize, self).__init__()
        self.mu, self.std = mu, std

    def forward(self, x):
        return (x - self.mu) / self.std


class IdentityLayer(nn.Module):
    def forward(self, inputs):
        return inputs


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(
        self, in_planes, planes, bn, learnable_bn, stride=1, activation="relu"
    ):
        super(PreActBlock, self).__init__()
        self.collect_preact = True
        self.activation = activation
        self.avg_preacts = []
        self.bn1 = (
            nn.BatchNorm2d(in_planes, affine=learnable_bn) if bn else IdentityLayer()
        )
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=not learnable_bn,
        )
        self.bn2 = (
            nn.BatchNorm2d(planes, affine=learnable_bn) if bn else IdentityLayer()
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=not learnable_bn
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=not learnable_bn,
                )
            )

    def act_function(self, preact):
        if self.activation == "relu":
            act = F.relu(preact)
        elif self.activation[:6] == "3prelu":
            act = tp_relu(preact, delta=float(self.activation.split("relu")[1]))
        elif self.activation[:8] == "3psmooth":
            act = tp_smoothed_relu(
                preact, delta=float(self.activation.split("smooth")[1])
            )
        else:
            assert self.activation[:8] == "softplus"
            beta = int(self.activation.split("softplus")[1])
            act = F.softplus(preact, beta=beta)
        return act

    def forward(self, x):
        out = self.act_function(self.bn1(x))
        shortcut = (
            self.shortcut(out) if hasattr(self, "shortcut") else x
        )  # Important: using out instead of x
        out = self.conv1(out)
        out = self.conv2(self.act_function(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        n_cls,
        device=True,
        half_prec=False,
        activation="relu",
        fts_before_bn=False,
        normal="none",
    ):
        super(PreActResNet, self).__init__()
        self.bn = True
        self.learnable_bn = True  # doesn't matter if self.bn=False
        self.in_planes = 64
        self.avg_preact = None
        self.activation = activation
        self.fts_before_bn = fts_before_bn
        if normal == "cifar10":
            self.mu = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
            self.std = torch.tensor((0.2471, 0.2435, 0.2616)).view(1, 3, 1, 1)
        else:
            self.mu = torch.tensor((0.0, 0.0, 0.0)).view(1, 3, 1, 1)
            self.std = torch.tensor((1.0, 1.0, 1.0)).view(1, 3, 1, 1)
            print("no input normalization")
        if device:
            self.mu = self.mu.to(device)
            self.std = self.std.to(device)
        if half_prec:
            self.mu = self.mu.half()
            self.std = self.std.half()

        self.normalize = Normalize(self.mu, self.std)
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=not self.learnable_bn
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512 * block.expansion, n_cls)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    self.bn,
                    self.learnable_bn,
                    stride,
                    self.activation,
                )
            )
            # layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        for layer in [*self.layer1, *self.layer2, *self.layer3, *self.layer4]:
            layer.avg_preacts = []

        out = self.normalize(x)
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if return_features and self.fts_before_bn:
            return out.view(out.size(0), -1)
        out = F.relu(self.bn(out))
        if return_features:
            return out.view(out.size(0), -1)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def PreActResNet18(
    n_cls,
    device="cpu",
    half_prec=False,
    activation="relu",
    fts_before_bn=False,
    normal="none",
):
    # print('initializing PA RN-18 with act {}, normal {}'.format())
    return PreActResNet(
        PreActBlock,
        [2, 2, 2, 2],
        n_cls=n_cls,
        device=device,
        half_prec=half_prec,
        activation=activation,
        fts_before_bn=fts_before_bn,
        normal=normal,
    )


"""# Implement the Attacks

Functions are given a simple useful signature that you can start with. Feel free to extend the signature as you see fit.

You may find it useful to create a 'batched' version of PGD that you can use to create the adversarial attack.
"""


def pgd_linf_untargeted(model, x, labels, k, eps, eps_step):
    model.eval()
    ce_loss = torch.nn.CrossEntropyLoss()
    adv_x = x.clone().detach()
    adv_x.requires_grad_(True)
    for _ in range(k):
        adv_x.requires_grad_(True)
        model.zero_grad()
        output = model(adv_x)
        # TODO: Calculate the loss
        loss = ce_loss(output, labels)
        loss.backward()
        with torch.no_grad():
            # TODO: compute the adv_x
            adv_x = adv_x + (eps_step * adv_x.grad.sign())
            # find delta, clamp with eps
            delta = adv_x - x
            delta = torch.clamp(delta, -eps, eps)
            adv_x = torch.clamp(x + delta, 0, 1)

    return adv_x


def pgd_l2_untargeted(model, x, labels, k, eps, eps_step):
    model.eval()
    ce_loss = torch.nn.CrossEntropyLoss()
    adv_x = x.clone().detach()
    adv_x.requires_grad_(True)
    for _ in range(k):
        adv_x.requires_grad_(True)
        model.zero_grad()
        output = model(adv_x)
        batch_size = x.size()[0]
        # TODO: Calculate the loss
        loss = ce_loss(output, labels)
        loss.backward()
        grad = adv_x.grad.data
        grad_norms = (
            torch.norm(grad.view(batch_size, -1), p=2, dim=1) + 1e-10
        )  # eps for avoiding div by 0
        grad_update = grad / grad_norms.view(batch_size, 1, 1, 1)
        # TODO: compute the adv_x
        adv_x = adv_x.detach() + eps_step * grad_update
        # find delta, clamp with eps, project delta to the l2 ball
        delta = adv_x - x
        delta_norm = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
        delta_norm = torch.min(eps / delta_norm, torch.ones_like(delta_norm))
        upd_delta = delta * delta_norm.view(batch_size, 1, 1, 1)
        adv_x = x + upd_delta

        # HINT: https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/pgdl2.py

    return adv_x


def test_model_standard(model, test_loader):
    model.eval()
    tot_test, tot_acc = 0.0, 0.0
    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            pred = torch.max(output, dim=1)[1]
            acc = (pred == y_batch).sum().item()
            tot_acc += acc
            tot_test += y_batch.size(0)
    standard_acc = tot_acc / tot_test
    return standard_acc


def test_model_on_single_attack(model, test_loader, attack="pgd_linf", eps=0.1, k=10, eps_step = None):
    model.eval()
    tot_test, tot_acc = 0.0, 0.0
    if eps_step is None:
        eps_step = eps / 4
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating")
    for batch_idx, (x_batch, y_batch) in pbar:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        if attack == "pgd_linf":
            # TODO: get x_adv untargeted pgd linf with eps, and eps_step=eps/4
            adv_x = pgd_linf_untargeted(model, x_batch, y_batch, k, eps, eps_step)

        elif attack == "pgd_l2":
            # TODO: get x_adv untargeted pgd l2 with eps, and eps_step=eps/4
            adv_x = pgd_l2_untargeted(model, x_batch, y_batch, k, eps, eps_step)
        else:
            raise NotImplementedError(f"Attack {attack} is not implemented")

        output = model(adv_x)
        pred = torch.max(output, dim=1)[1]
        acc = (pred == y_batch).sum().item()
        test = y_batch.size(0)

        # get the testing accuracy and update tot_test and tot_acc
        tot_acc += acc
        tot_test += test
        robustness_accuracy = tot_acc / tot_test
        pbar.set_postfix(robustness_accuracy=robustness_accuracy)

    print(f"Robust accuracy %.5lf" % (robustness_accuracy), f"on {attack} attack with eps={eps}")
    return robustness_accuracy


"""## Multi-Norm Robust Accuracy"""


def test_model_on_multi_attacks(model, test_loader, eps_linf=8.0 / 255.0, eps_l2=0.75):
    model.eval()
    tot_test, tot_acc = 0.0, 0.0
    for batch_idx, (x_batch, y_batch) in tqdm(
        enumerate(test_loader), total=len(test_loader), desc="Evaluating"
    ):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        # TODO: get x_adv_linf and x_adv_l2 untargeted pgd linf and l2 with eps, and eps_step=eps/4
        x_adv_linf = pgd_linf_untargeted(
            model, x_batch, y_batch, 4, eps_linf, eps_linf / 4
        )
        x_adv_l2 = pgd_l2_untargeted(model, x_batch, y_batch, 4, eps_l2, eps_l2 / 4)

        ## calculate union accuracy: correct only if both attacks are correct

        out = model(x_adv_linf)
        pred_linf = torch.max(out, dim=1)[1]
        out = model(x_adv_l2)
        pred_l2 = torch.max(out, dim=1)[1]
        acc = ((pred_linf == y_batch) & (pred_l2 == y_batch)).sum().item()
        test = y_batch.size(0)

        # TODO: get the testing accuracy with multi-norm robustness and update tot_test and tot_acc
        tot_acc += acc
        tot_test += test

    print("Robust accuracy %.5lf" % (tot_acc / tot_test), f"on multi attacks")


def hw1_test_multi_norm_robustness():
    ## Dataloaders
    train_dataset = datasets.CIFAR10(
        "cifar10_data/",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_dataset = datasets.CIFAR10(
        "cifar10_data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # intialize the model
    model = PreActResNet18(10, device=device, activation="softplus1").to(device)
    model.eval()

    """# Evaluate Single and Multi-Norm Robust Accuracy

    In this section, we evaluate the model on the Linf and L2 attacks as well as union accuracy.
    """
    """## Single-Norm Robust Accuracy"""

    # Evaluate on Linf attack with different models with eps = 8/255
    model.load_state_dict(torch.load("models/pretr_Linf.pth"))
    # Evaluate on Linf attack with model 1 with eps = 8/255
    test_model_on_single_attack(model, test_loader, attack="pgd_linf", eps=8.0 / 255.0)

    model.load_state_dict(torch.load("models/pretr_L2.pth"))
    # Evaluate on Linf attack with model 2 with eps = 8/255
    test_model_on_single_attack(model, test_loader, attack="pgd_l2", eps=8.0 / 255.0)

    model.load_state_dict(torch.load("models/pretr_RAMP.pth"))
    # Evaluate on Linf attack with model 3 with eps = 8/255
    test_model_on_single_attack(model, test_loader, attack="pgd_linf", eps=8.0 / 255.0)

    # Evaluate on L2 attack with different models with eps = 0.75
    model.load_state_dict(torch.load("models/pretr_Linf.pth"))
    test_model_on_single_attack(model, test_loader, attack="pgd_linf", eps=0.75)
    # Evaluate on Linf attack with model 1 with eps = 0.75

    model.load_state_dict(torch.load("models/pretr_L2.pth"))
    test_model_on_single_attack(model, test_loader, attack="pgd_l2", eps=0.75)
    # Evaluate on Linf attack with model 2 with eps = 0.75

    model.load_state_dict(torch.load("models/pretr_RAMP.pth"))
    test_model_on_single_attack(model, test_loader, attack="pgd_linf", eps=0.75)
    # Evaluate on Linf attack with model 3 with eps = 0.75
    # Evaluate on L2 attack with different models with eps = 0.5
    model.load_state_dict(torch.load("models/pretr_Linf.pth"))
    test_model_on_multi_attacks(model, test_loader, eps_linf=8.0 / 255.0, eps_l2=0.75)
    # Evaluate on multi attacks with model 1

    model.load_state_dict(torch.load("models/pretr_L2.pth"))
    test_model_on_multi_attacks(model, test_loader, eps_linf=8.0 / 255.0, eps_l2=0.75)
    # Evaluate on multi attacks with model 2

    model.load_state_dict(torch.load("models/pretr_RAMP.pth"))
    test_model_on_multi_attacks(model, test_loader, eps_linf=8.0 / 255.0, eps_l2=0.75)
    # Evaluate on multi attacks with model 3


### HW3 adversarial robustness training
def train_epoch(model, train_loader, optimizer):
    model.train()
    ce_loss = nn.CrossEntropyLoss()
    tot_loss, tot_acc = 0.0, 0.0
    tot_samples = 0
    pbar = tqdm(train_loader, desc="Training epoch..")
    for _, (x_batch, y_batch) in enumerate(pbar):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = ce_loss(logits, y_batch)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * x_batch.size(0)
        tot_acc += (logits.argmax(dim=1) == y_batch).sum().item()
        tot_samples += x_batch.size(0)
        pbar.set_postfix(loss=tot_loss / tot_samples, acc=tot_acc / tot_samples)
    avg_loss = tot_loss / tot_samples
    avg_acc = tot_acc / tot_samples
    return avg_loss, avg_acc


def standard_training(
    model, train_loader, test_loader, num_epochs=100, lr=0.1, save_path=None
):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[75, 90], gamma=0.1
    )
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_standard_acc": [],
    }
    print(f"\n{'=' * 60}")
    print(f"Starting Standard Training")
    print(f"{'=' * 60}\n")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
        )
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.5f}")
        if epoch % 1 == 0 or epoch == num_epochs:
            standard_acc = test_model_standard(model, test_loader)
            print(f"Standard Test Acc: {standard_acc:.5f}")
            history["test_standard_acc"].append(standard_acc)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        scheduler.step()
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}\n")

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"\nModel saved to {save_path}")

    return history


def adversarial_train_epoch(
    model, train_loader, optimizer, attack_type="linf", eps=8.0 / 255.0
):
    model.train()
    ce_loss = nn.CrossEntropyLoss()
    tot_loss, tot_acc = 0.0, 0.0
    tot_samples = 0
    pbar = tqdm(train_loader, desc="Training epoch.. ")
    for _, (x_batch, y_batch) in enumerate(pbar):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        # Generate adversarial examples
        model.eval()  # Set to eval mode for attack generation
        if attack_type == "pgd_linf":
            x_adv = pgd_linf_untargeted(
                model, x_batch, y_batch, k=10, eps=eps, eps_step=eps / 4
            )
        elif attack_type == "pgd_l2":
            x_adv = pgd_l2_untargeted(
                model, x_batch, y_batch, k=10, eps=eps, eps_step=eps / 4
            )
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
        # Train on adversarial examples
        x_adv = x_adv.detach()
        model.train()
        optimizer.zero_grad()
        output = model(x_adv)
        loss = ce_loss(output, y_batch)
        loss.backward()
        optimizer.step()
        # Calculate accuracy
        pred = torch.max(output, dim=1)[1]
        acc = (pred == y_batch).sum().item()
        tot_loss += loss.item() * y_batch.size(0)
        tot_acc += acc
        tot_samples += y_batch.size(0)
        pbar.set_postfix({"loss": tot_loss / tot_samples, "acc": tot_acc / tot_samples})
    avg_loss = tot_loss / tot_samples
    avg_acc = tot_acc / tot_samples
    return avg_loss, avg_acc


def adversarial_training(
    model,
    train_loader,
    test_loader,
    attack_type="linf",
    eps=2.0 / 255.0,
    num_epochs=5,
    lr=0.1,
    save_path=None,
):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[75, 90], gamma=0.1
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_standard_acc": [],
        "test_robust_acc": [],
    }

    print(f"\n{'=' * 60}")
    print(f"Starting Adversarial Training")
    print(f"Attack type: {attack_type}, Epsilon: {eps}")
    print(f"{'=' * 60}\n")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = adversarial_train_epoch(
            model, train_loader, optimizer, attack_type, eps
        )
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.5f}")
        if epoch % 1 == 0 or epoch == num_epochs:
            standard_acc = test_model_standard(model, test_loader)
            print(f"Standard Test Acc: {standard_acc:.5f}")
            robust_acc = test_model_on_single_attack(
                model, test_loader, attack_type, eps
            )
            print(f"Robust Test Acc: {robust_acc:.5f}")
            history["test_standard_acc"].append(standard_acc)
            history["test_robust_acc"].append(robust_acc)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        scheduler.step()
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}\n")

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"\nModel saved to {save_path}")

    return history


def evaluate_model_multiple_eps(
    model, test_loader, attack_type="linf", eps_values=[], save_dir=None
):
    model.eval()
    results = {}
    for eps in eps_values:
        print(f"Evaluating model with attack type {attack_type} and eps={eps}")
        if eps == 0.0:
            results[eps] = test_model_standard(model, test_loader)
        else:
            results[eps] = test_model_on_single_attack(
                model, test_loader, attack_type, eps
            )
        print(f"Test Acc: {results[eps]:.4f}")
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(
                save_dir,
                f"model_{model.__class__.__name__}_attacktype_{attack_type}_eps_{eps}.pth",
            )
            with open(save_path, "w") as f:
                json.dump(results, f)
    return results

def plot_history(history, tag="standard"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[0].set_xlabel("Epoch")
    axes[1].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{tag}_history.png")
    plt.close()
    print(f"✅ Saved training curve to plots/{tag}_history.png")

def plot_histories(histories, save_path=None, show=True):
    """
    Plot multiple training histories with markers for recorded values.

    Args:
        histories (dict): {name: history_dict}
            Each history_dict may contain:
              - 'train_loss'
              - 'train_acc'
              - 'test_standard_acc'
              - 'test_robust_acc' (optional)
        save_path (str): Optional path to save the figure.
        show (bool): Whether to display the figure (ignored if headless).
    """
    plt.style.use("seaborn-v0_8-darkgrid")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.reshape(2, 2)

    colors = plt.cm.tab10.colors
    markers = ["o", "s", "D", "^", "v", "x", "P", "*"]

    # ------------------ TRAIN LOSS ------------------
    for i, (label, hist) in enumerate(histories.items()):
        if "train_loss" in hist:
            epochs = range(1, len(hist["train_loss"]) + 1)
            axes[0, 0].plot(
                epochs, hist["train_loss"],
                label=label,
                color=colors[i % len(colors)],
                linestyle="-", linewidth=2,
                marker=markers[i % len(markers)],
                markersize=5, markeredgecolor="black", markeredgewidth=0.5,
            )
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")

    # ------------------ TRAIN ACCURACY ------------------
    for i, (label, hist) in enumerate(histories.items()):
        if "train_acc" in hist:
            epochs = range(1, len(hist["train_acc"]) + 1)
            axes[0, 1].plot(
                epochs, hist["train_acc"],
                label=label + " (Train)",
                color=colors[i % len(colors)],
                linestyle="-", linewidth=2,
                marker=markers[i % len(markers)],
                markersize=5, markeredgecolor="black", markeredgewidth=0.5,
            )
    axes[0, 1].set_title("Train Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")

    # ------------------ TEST STANDARD ACCURACY ------------------
    for i, (label, hist) in enumerate(histories.items()):
        if "test_standard_acc" in hist:
            epochs = range(1, len(hist["test_standard_acc"]) + 1)
            axes[1, 0].plot(
                epochs, hist["test_standard_acc"],
                label=label + " (Test)",
                color=colors[i % len(colors)],
                linestyle="-", linewidth=2,
                marker=markers[i % len(markers)],
                markersize=5, markeredgecolor="black", markeredgewidth=0.5,
            )
    axes[1, 0].set_title("Test Standard Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")

    # ------------------ TEST ROBUSTNESS ACCURACY ------------------
    for i, (label, hist) in enumerate(histories.items()):
        if "test_robust_acc" in hist:
            epochs = range(1, len(hist["test_robust_acc"]) + 1)
            axes[1, 1].plot(
                epochs, hist["test_robust_acc"],
                label=label + " (Robust)",
                color=colors[i % len(colors)],
                linestyle="-", linewidth=2,
                marker=markers[i % len(markers)],
                markersize=5, markeredgecolor="black", markeredgewidth=0.5,
            )
    axes[1, 1].set_title("Test Robustness Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")

    # ------------------ Common Styling ------------------
    all_axes = axes.flatten()
    max_epochs = max(len(h.get("train_loss", [])) for h in histories.values())
    for ax in all_axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, max_epochs + 1))

    fig.suptitle("Model Training & Robustness Comparison", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=250)
        print(f"✅ Saved comparison plot to {save_path}")

    # Show if possible
    if show:
        try:
            plt.show()
        except Exception:
            print("⚠️ Headless environment detected, skipping display.")

    plt.close(fig)

def hw3_adverserial_training(use_cached_model=False):
    ## Dataloaders
    train_dataset = datasets.CIFAR10(
        "cifar10_data/",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_dataset = datasets.CIFAR10(
        "cifar10_data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Train standard model (no adversarial training)
    print("\n" + "="*60)
    print("TRAINING STANDARD MODEL")
    print("="*60)
    model_reg = PreActResNet18(10, device=device, activation="softplus1").to(device)
    model_reg_save_path = "models/standard_trained.pth"
    model_reg_history_path = "logs/standard_trained/standard_trained_history.json"

    if use_cached_model and os.path.exists(model_reg_save_path):
        model_reg.load_state_dict(torch.load(model_reg_save_path))
        with open(model_reg_history_path, "r") as f:
            history = json.load(f)
        print("Loaded cached standard trained model")
    else:
        start_time = time.time()
        history = standard_training(
            model_reg,
            train_loader,
            test_loader,
            num_epochs=num_epochs,
            lr=learning_rate,
            save_path=model_reg_save_path,
        )
        end_time = time.time()
        print(f"Standard training time: {end_time - start_time:.2f} seconds")
        # Save history to logs directory
        os.makedirs(os.path.dirname(model_reg_history_path), exist_ok=True)
        with open(model_reg_history_path, "w") as f:
            json.dump(history, f)
        print("Standard training completed and saved")

    plot_history(history, tag="standard")

    # test_eps_linf = [2/255, 4/255, 8/255, 16/255 ]
    # test_eps_l2 = [0.1, 0.25, 0.5, 0.75]
    # print("Standard Model Robustness:")
    # standard_results_linf = evaluate_model_multiple_eps(
    #     model_reg, test_loader, "pgd_linf", test_eps_linf, None
    # )
    # standard_results_l2 = evaluate_model_multiple_eps(
    #     model_reg, test_loader, "pgd_l2", test_eps_l2, None
    # )
    # with open("logs/standard_model_robustness.json", "w") as f:
    #     json.dump(
    #         {
    #             "model_weights": model_reg_save_path,
    #             "test_eps_linf": test_eps_linf,
    #             "test_eps_l2": test_eps_l2,
    #             "pgd_linf": standard_results_linf,
    #             "pgd_l2": standard_results_l2,
    #         },
    #         f,
    #     )


    # Train adversarially trained models with different epsilon values
    epsilon_values = [2/255]  # Different perturbation strengths

    for eps in epsilon_values:
        print(f"\n" + "="*60)
        print(f"ADVERSARIAL TRAINING with ε = {eps:.4f}")
        print("="*60)

        model_adv = PreActResNet18(10, device=device, activation="softplus1").to(device)
        model_adv_save_path = f"models/adv_trained_linf_eps_{int(eps*255)}.pth"
        model_adv_history_path = f"logs/adv_trained_linf/adv_trained_linf_eps_{int(eps*255)}_history.json"

        if use_cached_model and os.path.exists(model_adv_save_path):
            model_adv.load_state_dict(torch.load(model_adv_save_path))
            with open(model_adv_history_path, "r") as f:
                history_adv = json.load(f)
            print(f"Loaded cached adversarially trained model (ε={eps:.4f})")
        else:
            start_time = time.time()
            history_adv = adversarial_training(
                model_adv,
                train_loader,
                test_loader,
                attack_type="pgd_linf",
                eps=eps,
                num_epochs=num_epochs,
                lr=learning_rate,
                save_path=model_adv_save_path,
            )
            end_time = time.time()
            print(f"Adversarial training time (ε={eps:.4f}): {end_time - start_time:.2f} seconds")
            # Save history to logs directory
            os.makedirs(os.path.dirname(model_adv_history_path), exist_ok=True)
            with open(model_adv_history_path, "w") as f:
                json.dump(history_adv, f)
            print(f"Adversarial training completed and saved (ε={eps:.4f})")

        plot_history(history_adv, tag=f"adv_trained_linf_eps_{int(eps*255)}")

        # print("Adversarially Trained Model Robustness:")
        # adv_results_linf = evaluate_model_multiple_eps(
        #     model_adv, test_loader, "pgd_linf", test_eps_linf, None
        # )
        # adv_results_l2 = evaluate_model_multiple_eps(
        #     model_adv, test_loader, "pgd_l2", test_eps_l2, None
        # )

        # test_results_path = f"logs/adv_trained_linf_{int(eps*255)}_robustness.json"
        # with open(test_results_path, "w") as f:
        #     json.dump(
        #         {
        #             "model_weights": model_adv_save_path,
        #             "training_epsilon": eps,
        #             "test_eps_linf": test_eps_linf,
        #             "test_eps_l2": test_eps_l2,
        #             "pgd_linf": adv_results_linf,
        #             "pgd_l2": adv_results_l2,
        #         },
        #         f,
        #     )
    
    plot_histories(
        {
            "Standard Training": history,
            "Adv Training ε=2/255": history_adv,
        },
        save_path="plots/hw3_training_comparison.png",
        show=False,
    )

def hw3_attack_robustness_over_fgsm_attacks():
    test_dataset = datasets.CIFAR10(
        "cifar10_data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    model = PreActResNet18(10, device=device, activation="softplus1").to(device)

    saved_models_list = [f"adv_trained_linf_eps_{int(eps)}" for eps in [2]] + [
        "standard_trained",
        "pretr_Linf",
        "pretr_L2",
        "pretr_RAMP",
    ]

    linf_eps_values = [1 / 255, 2 / 255, 4 / 255, 8 / 255, 16 / 255]
    l2_eps_values = [0.1, 0.25, 0.5, 0.75, 1.0]

    all_results = {}

    for saved_weights in saved_models_list:
        print(f"Processing {saved_weights}")
        model.load_state_dict(torch.load(f"models/{saved_weights}.pth"))
        model.eval()
        results = defaultdict(dict)

        print(f"Testing {saved_weights} on FGSM-Linf")
        for eps in tqdm(linf_eps_values):
            results["fgsm_linf"][eps] = test_model_on_single_attack(
                model, test_loader, "pgd_linf", eps, k=1, eps_step = eps
            )
        print(f"Testing {saved_weights} on FGSM-L2")
        for eps in tqdm(l2_eps_values):
            results["fgsm_l2"][eps] = test_model_on_single_attack(
                model, test_loader, "pgd_l2", eps, k=1, eps_step = eps
            )

        all_results[saved_weights] = results

        with open("logs/fgsm_attack_results.json", "w") as f:
            json.dump(all_results, f)

    return all_results


if __name__ == "__main__":
    hw3_adverserial_training(use_cached_model=True)
    # hw3_attack_robustness_over_fgsm_attacks()
