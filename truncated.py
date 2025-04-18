import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



class SyntheticPiecewiseDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=20, d=3, a=-25.0, b=25.0, seed=42):

        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.data = 10 * torch.rand(num_samples, seq_len, d) - 5
        self.seq_len = seq_len
        self.d = d
        self.a = a
        self.b = b
        # self.data = torch.randn(num_samples, seq_len, d)
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
        self.modulation = nn.Parameter(torch.ones(seq_len, hidden_dim)) # v_i
        extra = p - seq_len
        self.interp_tokens = nn.Parameter(torch.linspace(-1.0, 1.0, extra)
                                          .unsqueeze(1).expand(extra, hidden_dim).clone())

    def forward(self, x):
        batch_size = x.size(0)
        token_repr = self.token_proj(x)                     # (batch, seq_len, hidden_dim)
        token_repr = token_repr * self.modulation.unsqueeze(0)  # (batch, seq_len, hidden_dim)
        interp_repr = self.interp_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, extra, hidden_dim)
        out = torch.cat([token_repr, interp_repr], dim=1)    # (batch, p, hidden_dim)
        return out

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
 
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        self.q_proj = nn.Linear(input_dim, hidden_dim)
        self.k_proj = nn.Linear(input_dim, hidden_dim)
        self.v_proj = nn.Linear(input_dim, output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)

    def forward(self, x, return_weights=False):
        batch_size, seq_len, _ = x.size()
        Q = self.q_proj(x)  
        K = self.k_proj(x)  
        V = self.v_proj(x) 

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        scores = torch.matmul(Q, K.transpose(-2,-1)) / (self.head_dim**0.5)  # (batch, num_heads, seq_len, seq_len)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (batch, num_heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        out = self.out_proj(attn_output)
        if return_weights:
            return out, attn_weights
        return out


class MultiHeadAttentionModelExtended(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len=20, p=30, num_heads=1):

        super().__init__()
        self.seq_len = seq_len
        self.p = p
        self.mapping_A = ExtendedMappingA(input_dim=input_dim, hidden_dim=hidden_dim, seq_len=seq_len, p=p)
        self.attn_layer = MultiHeadAttentionLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, num_heads=num_heads)

    # def forward(self, x, return_weights=False):
    #     x_mapped = self.mapping_A(x)  # (batch, p, hidden_dim)
    #     out, attn_weights = self.attn_layer(x_mapped, return_weights=True)  # (batch, p, hidden_dim), (batch, num_heads, p, p)
    #     actual_out = out[:, :self.seq_len, :]  # (batch, seq_len, hidden_dim)
    #     weights_pool = F.softmax(actual_out, dim=-1)
    #     pooled = torch.sum(actual_out * weights_pool, dim=-1, keepdim=True)  # (batch, seq_len, 1)
    #     if return_weights:
    #         return pooled, attn_weights
    #     return pooled
    def forward(self, x, return_weights=False):

        x_mapped = self.mapping_A(x)               
        out, attn_weights = self.attn_layer(x_mapped, return_weights=True)
        actual_out = out[:, :self.seq_len, :]        
        if return_weights:
            return actual_out, attn_weights
        return actual_out

def run_experiment_for_head(seed, num_heads, num_epochs=50, batch_size=32, lr=1e-3, seq_len=20, p=30, input_dim=3, a=-25.0, b=25.0):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset = SyntheticPiecewiseDataset(num_samples=1000, seq_len=seq_len, d=input_dim, a=a, b=b, seed=seed)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MultiHeadAttentionModelExtended(input_dim=input_dim, hidden_dim=32, seq_len=seq_len, p=p, num_heads=num_heads).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            # loss = criterion(pred, batch_y)
            # loss.backward()
            # optimizer.step()
            true = torch.zeros_like(pred)
            true[:, :, 0:1] = batch_y  
            loss = criterion(pred, true)
            loss.backward()
            optimizer.step()
    #         epoch_loss += loss.item() * batch_x.size(0)
    #     epoch_loss /= len(dataset)
    #     train_losses.append(epoch_loss)
    #     if (epoch + 1) % 20 == 0:
    #         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    # return model, train_losses
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            true = torch.zeros_like(pred)
            true[:, :, 0:1] = batch_y  
            loss = criterion(pred, true)
            # loss.backward()
            # optimizer.step()
            # loss = criterion(pred, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            count += batch_x.size(0)
    test_loss = total_loss / count
    return test_loss

#### head experiment
seeds = [1234,1235,1236,1237,1238,1239,1240,1241,1242,1243,1244]
# seeds = [1234,1235,1236]
head_list = [1, 2, 4, 8, 16]

file_path = "multihead_experiment_results.csv"
if not os.path.isfile(file_path):
    pd.DataFrame(columns=["seed", "num_heads", "test_loss"]).to_csv(file_path, index=False)
results = []
for num_heads in head_list:
    for seed in seeds:
        loss = run_experiment_for_head(seed, num_heads, num_epochs=50, batch_size=32, lr=1e-3, seq_len=50, p=60, input_dim=10, a=-25, b=25)
        results.append({"seed": seed, "num_heads": num_heads, "test_loss": loss})
        print(f"Seed {seed}, num_heads {num_heads}: Test Loss = {loss:.4f}")
        df = pd.DataFrame([results[-1]])
        df.to_csv(file_path, mode='a', header=not os.path.isfile(file_path), index=False)
df = pd.read_csv("multihead_experiment_results.csv")

grouped = df.groupby("num_heads")["test_loss"].agg(["mean", "std"]).reset_index()
print(grouped)


sns.set_context("notebook", font_scale=1.4, rc={"lines.linewidth": 2.5})
sns.set_style("whitegrid")

plt.figure(figsize=(8, 8))
plt.plot(grouped["num_heads"], grouped["mean"], marker="o", color="#1b7837")
plt.fill_between(grouped["num_heads"],
                 grouped["mean"] - grouped["std"],
                 grouped["mean"] + grouped["std"],
                 color="#1b7837", alpha=0.3)
plt.xlabel("Number of Heads", fontsize=25)
plt.ylabel("Test Accuracy (MSE)", fontsize=25)
plt.grid(True, linestyle='--', linewidth=0.6)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig("head_d50_gaussian.pdf", bbox_inches="tight")
plt.show()

#### |b-a| experiment
seeds = [1234,1235,1236,1237,1238,1239,1240,1241,1242,1243,1244]
# seeds = [1234,1235,1236]
range_list = [10,20,30,40,50]
head_list = [1]
file_path = "range_exp_new.csv"
if not os.path.isfile(file_path):
    pd.DataFrame(columns=["seed", "range", "test_loss"]).to_csv(file_path, index=False)
results = []
for num_heads in head_list:
    for r in range_list:
        a = -r/2
        b = r/2
        for seed in seeds:
            loss = run_experiment_for_head(seed, num_heads, num_epochs=50, batch_size=32, lr=1e-3, seq_len=50, p=60, input_dim=10, a=a, b=b)
            results.append({"seed": seed, "range": r, "test_loss": loss})
            print(f"Seed {seed}, range {r}: Test Loss = {loss:.4f}")
            df = pd.DataFrame([results[-1]])
            df.to_csv(file_path, mode='a', header=not os.path.isfile(file_path), index=False)

df = pd.read_csv(file_path)
df =df[df["range"] != 60]
grouped = df.groupby("range")["test_loss"].agg(["mean", "std"]).reset_index()
print(grouped)

sns.set_context("notebook", font_scale=1.4, rc={"lines.linewidth": 2.5})
sns.set_style("whitegrid")

plt.figure(figsize=(8, 8))
plt.plot(grouped["range"], grouped["mean"], marker="o", color="#1b7837", label="d=50")
plt.fill_between(grouped["range"],
                 grouped["mean"] - grouped["std"],
                 grouped["mean"] + grouped["std"],
                 color="#1b7837", alpha=0.3)
plt.xlabel(r"$|b-a|$", fontsize=25)
plt.ylabel("Test Accuracy (MSE)", fontsize=25)
plt.grid(True, linestyle='--', linewidth=0.6)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig("b_a_d10.pdf", bbox_inches="tight")
plt.show()

# p experiment
seeds = [1234,1235,1236,1237,1238,1239,1240,1241,1242,1243,1244]
p_list = [60,70,80,90,100,110,120]
head_list = [1]
file_path = "p_experiment_results_new_3.csv"
if not os.path.isfile(file_path):
    pd.DataFrame(columns=["seed", "p", "test_loss"]).to_csv(file_path, index=False)
results = []
for num_heads in head_list:
    for p in p_list:
        a = -25
        b = 25
        for seed in seeds:
            loss = run_experiment_for_head(seed, num_heads, num_epochs=50, batch_size=32, lr=1e-3, seq_len=50, p=p, input_dim=10, a=a, b=b)
            results.append({"seed": seed, "p": p, "test_loss": loss})
            print(f"Seed {seed}, p {p}: Test Loss = {loss:.4f}")
            df = pd.DataFrame([results[-1]])
            df.to_csv(file_path, mode='a', header=not os.path.isfile(file_path), index=False)

df = pd.DataFrame(results)

results = pd.read_csv("p_experiment_results_new_3.csv")
df = pd.DataFrame(results)
grouped = df.groupby("p")["test_loss"].agg(["mean", "std"]).reset_index()
print(grouped)

sns.set_context("notebook", font_scale=1.4, rc={"lines.linewidth": 2.5})
sns.set_style("whitegrid")



plt.figure(figsize=(8, 8))

plt.plot(grouped["p"], grouped["mean"], marker="o", color="#1b7837", label="d=50")
plt.fill_between(grouped["p"],
                 grouped["mean"] - grouped["std"],
                 grouped["mean"] + grouped["std"],
                 color="#1b7837", alpha=0.3)
plt.xlabel("p", fontsize=25)
plt.ylabel("Test Accuracy (MSE)", fontsize=25)
plt.grid(True, linestyle='--', linewidth=0.6)
plt.legend(fontsize=20)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig("p_d50_new.pdf", bbox_inches="tight")
plt.show()