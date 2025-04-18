import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class SyntheticPiecewiseDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=20, d=1, a=-1.0, b=1.0, seed=1234):
        super().__init__()
        self.data = 10 * torch.rand(num_samples, seq_len, d) - 5
        self.seq_len = seq_len
        self.d = d
        self.a = a
        self.b = b
        self.w = torch.randn(1, seq_len, d)  # shape: (1, seq_len, d)
        self.t = torch.randn(1, seq_len, 1)  # shape: (1, seq_len, 1)
        self.targets = self.truncated_linear(self.data)

    def truncated_linear(self, x):
        y = torch.sum(self.w * x, dim=-1, keepdim=True) + self.t
        return torch.clamp(y, min=self.a, max=self.b)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class ExtendedMappingA(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len=20, p=30):
        super().__init__()
        self.seq_len = seq_len
        self.p = p
        self.hidden_dim = hidden_dim

        self.token_proj = nn.Linear(input_dim, hidden_dim)
        self.modulation = nn.Parameter(torch.ones(seq_len, hidden_dim))
        extra = p - seq_len
        self.interp_tokens = nn.Parameter(torch.linspace(-1.0, 1.0, extra)
                                          .unsqueeze(1).expand(extra, hidden_dim).clone())

    def forward(self, x):
        batch_size = x.size(0)
        token_repr = self.token_proj(x)                    
        token_repr = token_repr * self.modulation.unsqueeze(0)  
        interp_repr = self.interp_tokens.unsqueeze(0).expand(batch_size, -1, -1)  
        out = torch.cat([token_repr, interp_repr], dim=1)
        return out

class SingleHeadAttentionLayerWithManualKeyValue(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, p, a=-25.0, b=25.0, beta=25, t_val=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.p = p
        self.beta = beta  
        self.t_val = t_val
        self.q_proj = nn.Linear(input_dim, hidden_dim)
        self.k_proj = nn.Linear(input_dim, hidden_dim)
        self.v_proj = nn.Linear(input_dim, output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)
        self.register_buffer("interp_full_K", torch.linspace(a, b, p).unsqueeze(1))
        self.register_buffer("interp_full_V", torch.linspace(a, b, p).unsqueeze(1))
        self.v_override_index = random.randint(0, output_dim - 1)

    def forward(self, x, return_weights=False):
        Q = self.q_proj(x) 
        K = self.k_proj(x)  
        V = self.v_proj(x)  
        k_range = torch.arange(0, self.p, device=x.device).float().unsqueeze(1) 
        new_val = (k_range * (self.interp_full_K + self.interp_full_K[0]) - 2 * k_range * self.t_val)
        K[:, :, -1] = new_val.unsqueeze(0).expand(K.size(0), -1, -1).squeeze(-1)
        V[:, :, self.v_override_index] = self.interp_full_V.to(V.device).unsqueeze(0).expand(V.size(0), -1, -1).squeeze(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        scores = self.beta * scores
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        out = self.out_proj(attn_output)
        if return_weights:
            return out, attn_weights
        return out


class OneLayerAttentionModelExtendedManualKV(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len=20, p=30, a=-25.0, b=25.0):
        super().__init__()
        self.seq_len = seq_len
        self.p = p
        self.mapping_A = ExtendedMappingA(input_dim=input_dim, hidden_dim=hidden_dim, seq_len=seq_len, p=p)
        self.attn_layer = SingleHeadAttentionLayerWithManualKeyValue(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, p=p, a=a, b=b)

    def forward(self, x, return_weights=False):
        x_mapped = self.mapping_A(x)   
        out, attn_weights = self.attn_layer(x_mapped, return_weights=True)
        actual_out = out[:, :self.seq_len, :]
        return actual_out, attn_weights


def train_model_one_layer_manual(train_loader, num_epochs=100, batch_size=32, lr=1e-3, seq_len=30, p=30, a=-25, b=25, input_dim=1):
    dataloader = train_loader
    model = OneLayerAttentionModelExtendedManualKV(input_dim=input_dim, hidden_dim=11, seq_len=seq_len, p=p, a=a, b=b).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_losses = []
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred, _ = model(batch_x)
            true = torch.zeros_like(pred)
            override_idx = model.attn_layer.v_override_index
            true[:, :, override_idx:override_idx+1] = batch_y
            loss = criterion(pred, true)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        epoch_loss /= len(dataloader)
        train_losses.append(epoch_loss)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    return model, train_losses

def plot_training_loss(train_losses):
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 4))
    sns.scatterplot(x=list(range(len(train_losses))), y=train_losses, s=100, label="Training Loss")
    plt.xlabel("Epoch", fontsize=25)
    plt.ylabel("MSE Loss", fontsize=25)
    plt.title("Training Loss Curve (Manual Key & Value Overrides)", fontsize=25)
    plt.legend(fontsize=20)
    plt.show()




def plot_attention_grid_for_runs(num_examples=3):


    r_values = [0.5,1.5,3]
    # r_values = [1,2,3]
    # r_values = [1,2,3,4,20]

    attn_results = {}
    for r in r_values:
        seed = 1234
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        a = -r/2
        b = r/2
        print(f"Running for r = {r} (a={a}, b={b})")
        dataset = SyntheticPiecewiseDataset(num_samples=1000, seq_len=seq_len, d=input_dim, a=a, b=b, seed=seed)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        batch_x, _ = next(iter(test_loader))
        batch_x = batch_x.to(device)

        batch_x_sample = batch_x[:num_examples]

        model_r, _ = train_model_one_layer_manual(train_loader, num_epochs=100, batch_size=32, lr=1e-3,
                                                    seq_len=30, p=30, a=a, b=b, input_dim=input_dim)
        model_r.eval()
        with torch.no_grad():
            _, attn_weights = model_r(batch_x_sample, return_weights=True)

        attn_results[r] = attn_weights.cpu().numpy()


    fig, axes = plt.subplots(len(r_values), num_examples, figsize=(4 * num_examples, 4 * len(r_values)))
    for i, r in enumerate(r_values):
        for j in range(num_examples):
            sample_weights = attn_results[r][j].T 
            sns.heatmap(sample_weights, cmap=sns.cubehelix_palette(as_cmap=True), ax=axes[i, j], cbar=False)
            axes[i, j].invert_yaxis()
            axes[i, j].set_title(f"|b-a|={r}, Sample {j}", fontsize=16)
            axes[i, j].set_xlabel("Query Token Index", fontsize=14)
            axes[i, j].set_ylabel("Key Token Index", fontsize=14)
    plt.tight_layout()
    plt.savefig("attention_grid_runs_1000.pdf", bbox_inches="tight")
    plt.show()


input_dim = 10   
r_val = 20      
seq_len = 30
batch_size = 32
p = 30

plot_attention_grid_for_runs(num_examples=3)
