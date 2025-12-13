# =============================================================================
# MNIST Superposition Experiment
# Extension of Anthropic's Toy Models of Superposition to MNIST
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import trange, tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# =============================================================================
# 1. Load MNIST Dataset
# =============================================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('../hw2/mnist_data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('../hw2/mnist_data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# =============================================================================
# 2. Define Autoencoder with Bottleneck (to study superposition)
# =============================================================================

class MNISTAutoencoder(nn.Module):
    """
    Autoencoder with a bottleneck layer.
    Input: 784 (28x28 flattened)
    Architecture: 784 -> 256 -> 64 -> bottleneck_dim -> 64 -> 256 -> 784
    
    We study superposition at the bottleneck layer.
    """
    def __init__(self, bottleneck_dim=16):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        
        # Encoder
        self.enc1 = nn.Linear(784, 256)
        self.enc2 = nn.Linear(256, 64)
        self.enc3 = nn.Linear(64, bottleneck_dim)  # Bottleneck
        
        # Decoder
        self.dec1 = nn.Linear(bottleneck_dim, 64)
        self.dec2 = nn.Linear(64, 256)
        self.dec3 = nn.Linear(256, 784)
        
    def encode(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = self.enc3(x)  # No activation at bottleneck
        return x
    
    def decode(self, z):
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = torch.sigmoid(self.dec3(x))
        return x.view(-1, 1, 28, 28)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

# =============================================================================
# 3. Training Function
# =============================================================================

def train_autoencoder(model, train_loader, epochs=20, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    losses = []
    for epoch in trange(epochs, desc="Training"):
        epoch_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon, z = model(data)
            loss = F.mse_loss(recon, data)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")
    
    return losses

# =============================================================================
# 4. Analysis Functions for Superposition
# =============================================================================

def compute_feature_activations(model, data_loader, n_batches=50):
    """Compute activations at the bottleneck for analysis."""
    model.eval()
    activations = []
    
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            if i >= n_batches:
                break
            data = data.to(device)
            z = model.encode(data)
            activations.append(z.cpu())
    
    return torch.cat(activations, dim=0)

def analyze_superposition(model, data_loader):
    """
    Analyze superposition in the bottleneck layer.
    We look at:
    1. Effective dimensionality of representations
    2. Weight matrix structure (W^T W for encoder)
    3. Activation sparsity
    """
    model.eval()
    
    # Get encoder weight to bottleneck
    W = model.enc3.weight.detach().cpu()  # [bottleneck_dim, 64]
    
    # Compute W^T W
    WtW = W @ W.T  # [bottleneck_dim, bottleneck_dim]
    
    # Compute activations
    activations = compute_feature_activations(model, data_loader)
    
    # Activation statistics
    act_mean = activations.mean(dim=0)
    act_std = activations.std(dim=0)
    
    # Sparsity: fraction of near-zero activations
    sparsity = (activations.abs() < 0.1).float().mean(dim=0)
    
    # Correlation matrix of activations
    act_centered = activations - act_mean
    cov = (act_centered.T @ act_centered) / (activations.shape[0] - 1)
    std_outer = act_std.unsqueeze(0) * act_std.unsqueeze(1)
    corr = cov / (std_outer + 1e-8)
    
    # Effective dimensionality via participation ratio
    _, S, _ = torch.svd(activations - activations.mean(dim=0))
    S_norm = S / S.sum()
    participation_ratio = 1.0 / (S_norm ** 2).sum()
    
    return {
        'W': W,
        'WtW': WtW,
        'activations': activations,
        'sparsity': sparsity,
        'correlation': corr,
        'singular_values': S,
        'participation_ratio': participation_ratio.item(),
        'act_mean': act_mean,
        'act_std': act_std
    }

def analyze_decoder_superposition(model):
    """
    Analyze if decoder represents more 'output features' than bottleneck dims.
    Look at first decoder layer: bottleneck_dim -> 64
    """
    W_dec = model.dec1.weight.detach().cpu()  # [64, bottleneck_dim]
    
    # Each row is how one output unit is composed from bottleneck
    # Compute interference: dot products between output feature directions
    W_norm = W_dec / (W_dec.norm(dim=1, keepdim=True) + 1e-8)
    interference = W_norm @ W_norm.T  # [64, 64]
    
    # Off-diagonal interference magnitude
    mask = 1 - torch.eye(64)
    off_diag = (interference * mask).abs()
    
    return {
        'W_dec': W_dec,
        'interference': interference,
        'mean_interference': off_diag.mean().item(),
        'max_interference': off_diag.max().item()
    }

# =============================================================================
# 5. Visualization Functions
# =============================================================================

def plot_superposition_analysis(analysis, decoder_analysis, bottleneck_dim):
    """Create visualization similar to toy model paper."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. W^T W matrix (encoder bottleneck)
    ax = axes[0, 0]
    im = ax.imshow(analysis['WtW'].numpy(), cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title(f'Encoder $W^T W$ (bottleneck={bottleneck_dim})')
    ax.set_xlabel('Bottleneck dimension')
    ax.set_ylabel('Bottleneck dimension')
    plt.colorbar(im, ax=ax)
    
    # 2. Decoder interference matrix
    ax = axes[0, 1]
    im = ax.imshow(decoder_analysis['interference'].numpy(), cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title(f'Decoder Interference (64 features in {bottleneck_dim} dims)')
    ax.set_xlabel('Output feature')
    ax.set_ylabel('Output feature')
    plt.colorbar(im, ax=ax)
    
    # 3. Singular value spectrum
    ax = axes[0, 2]
    S = analysis['singular_values'].numpy()
    ax.bar(range(len(S)), S / S.max())
    ax.axhline(y=0.1, color='r', linestyle='--', label='10% threshold')
    ax.set_title(f'Singular Values (PR={analysis["participation_ratio"]:.1f})')
    ax.set_xlabel('Component')
    ax.set_ylabel('Normalized singular value')
    ax.legend()
    
    # 4. Activation sparsity per dimension
    ax = axes[1, 0]
    ax.bar(range(bottleneck_dim), analysis['sparsity'].numpy())
    ax.set_title('Activation Sparsity per Bottleneck Dim')
    ax.set_xlabel('Bottleneck dimension')
    ax.set_ylabel('Fraction near-zero')
    ax.set_ylim(0, 1)
    
    # 5. Activation correlation matrix
    ax = axes[1, 1]
    im = ax.imshow(analysis['correlation'].numpy(), cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title('Activation Correlation Matrix')
    ax.set_xlabel('Bottleneck dimension')
    ax.set_ylabel('Bottleneck dimension')
    plt.colorbar(im, ax=ax)
    
    # 6. Feature norms (decoder weights)
    ax = axes[1, 2]
    norms = decoder_analysis['W_dec'].norm(dim=1).numpy()
    colors = ['blue' if decoder_analysis['interference'][i, :].abs().mean() < 0.3 
              else 'orange' for i in range(64)]
    ax.bar(range(64), norms, color=colors)
    ax.axvline(x=bottleneck_dim - 0.5, color='black', linestyle='-', linewidth=2)
    ax.set_title('Decoder Feature Norms (orange=polysemantic)')
    ax.set_xlabel('Output feature index')
    ax.set_ylabel('Norm')
    
    plt.tight_layout()
    return fig

def plot_reconstructions(model, test_loader, n_samples=10):
    """Visualize original vs reconstructed images."""
    model.eval()
    data, _ = next(iter(test_loader))
    data = data[:n_samples].to(device)
    
    with torch.no_grad():
        recon, _ = model(data)
    
    fig, axes = plt.subplots(2, n_samples, figsize=(15, 3))
    for i in range(n_samples):
        axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
    
    axes[0, 0].set_title('Original', loc='left')
    axes[1, 0].set_title('Reconstructed', loc='left')
    plt.tight_layout()
    return fig

# =============================================================================
# 6. Run Experiments with Different Bottleneck Sizes
# =============================================================================

def run_experiment(bottleneck_dim, epochs=20):
    """Run full experiment for a given bottleneck dimension."""
    print(f"\n{'='*60}")
    print(f"Training with bottleneck_dim = {bottleneck_dim}")
    print(f"{'='*60}")
    
    model = MNISTAutoencoder(bottleneck_dim=bottleneck_dim).to(device)
    losses = train_autoencoder(model, train_loader, epochs=epochs)
    
    # Analyze
    analysis = analyze_superposition(model, test_loader)
    decoder_analysis = analyze_decoder_superposition(model)
    
    print(f"\nResults for bottleneck_dim = {bottleneck_dim}:")
    print(f"  Participation ratio: {analysis['participation_ratio']:.2f}")
    print(f"  Mean decoder interference: {decoder_analysis['mean_interference']:.4f}")
    print(f"  Max decoder interference: {decoder_analysis['max_interference']:.4f}")
    
    # Plot
    fig = plot_superposition_analysis(analysis, decoder_analysis, bottleneck_dim)
    plt.savefig(f'superposition_analysis_b{bottleneck_dim}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    fig_recon = plot_reconstructions(model, test_loader)
    plt.savefig(f'reconstructions_b{bottleneck_dim}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return model, analysis, decoder_analysis

# =============================================================================
# 7. Main Experiment: Compare Different Bottleneck Sizes
# =============================================================================

# Run experiments with varying bottleneck sizes
# Smaller bottleneck = more pressure for superposition
results = {}

for bottleneck_dim in [16, 32, 64]:
    model, analysis, decoder_analysis = run_experiment(bottleneck_dim, epochs=20)
    results[bottleneck_dim] = {
        'model': model,
        'analysis': analysis,
        'decoder_analysis': decoder_analysis
    }

# =============================================================================
# 8. Summary Comparison Plot
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

bottleneck_dims = list(results.keys())

# Participation ratio vs bottleneck size
ax = axes[0]
prs = [results[b]['analysis']['participation_ratio'] for b in bottleneck_dims]
ax.bar(range(len(bottleneck_dims)), prs, tick_label=bottleneck_dims)
ax.set_xlabel('Bottleneck Dimension')
ax.set_ylabel('Participation Ratio')
ax.set_title('Effective Dimensionality')

# Mean interference vs bottleneck size
ax = axes[1]
interf = [results[b]['decoder_analysis']['mean_interference'] for b in bottleneck_dims]
ax.bar(range(len(bottleneck_dims)), interf, tick_label=bottleneck_dims)
ax.set_xlabel('Bottleneck Dimension')
ax.set_ylabel('Mean Interference')
ax.set_title('Decoder Feature Interference')

# Interpretation text
ax = axes[2]
ax.axis('off')
text = """
Superposition Indicators:

1. High interference (off-diagonal in W^TW)
   suggests features share dimensions

2. Participation ratio < bottleneck dim
   suggests some dimensions unused
   
3. Polysemantic features (orange bars)
   indicate superposition

Key Finding: Smaller bottlenecks force
more superposition to maintain 
reconstruction quality.
"""
ax.text(0.1, 0.5, text, fontsize=11, verticalalignment='center', fontfamily='monospace')

plt.tight_layout()
plt.savefig('superposition_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("EXPERIMENT COMPLETE")
print("="*60)