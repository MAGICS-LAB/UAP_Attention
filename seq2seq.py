import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class NPZDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        if 'X_train' in data:
            self.X = data['X_train']
            self.Y = data['Y_train']
        else:
            self.X = data['X']
            self.Y = data['Y']
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class ExtendedMappingA(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len=20, p=30):
        super().__init__()
        self.seq_len = seq_len
        self.p = p
        self.hidden_dim = hidden_dim
        self.token_proj = nn.Linear(input_dim, hidden_dim)
        self.modulation = nn.Parameter(torch.ones(seq_len, hidden_dim))
        extra = p - seq_len
        self.interp_tokens = nn.Parameter(
            torch.linspace(-1.0, 1.0, extra).unsqueeze(1).expand(extra, hidden_dim).clone()
        )

    def forward(self, x):
        batch_size = x.size(0)
        token_repr = self.token_proj(x)                    
        token_repr = token_repr * self.modulation.unsqueeze(0)  
        interp_repr = self.interp_tokens.unsqueeze(0).expand(batch_size, -1, -1)  
        out = torch.cat([token_repr, interp_repr], dim=1)   
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
        scores = torch.matmul(Q, K.transpose(-2,-1)) / (self.head_dim**0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        out = self.out_proj(attn_output)
        if return_weights:
            return out, attn_weights
        return out

class MultiLayerAttentionModelExtended(nn.Module):
    def __init__(self, input_dim, hidden_dim, target_dim, seq_len=20, p=30, num_heads=1, num_layers=2):

        super().__init__()
        self.seq_len = seq_len
        self.p = p

        self.mapping_A = ExtendedMappingA(input_dim=input_dim, hidden_dim=hidden_dim, seq_len=seq_len, p=p)
        self.attn1 = MultiHeadAttentionLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                              output_dim=hidden_dim, num_heads=num_heads)
        self.A2 = nn.Linear(hidden_dim, hidden_dim)
        self.attn2 = MultiHeadAttentionLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                              output_dim=hidden_dim, num_heads=num_heads)
        self.out_proj = nn.Linear(hidden_dim, target_dim)

    def forward(self, x, return_weights=False):

        x_extended = self.mapping_A(x)  
        x_attn1, attn1_weights = self.attn1(x_extended, return_weights=True)
        x_A2 = self.A2(x_attn1) 

        x_attn2, attn2_weights = self.attn2(x_A2, return_weights=True)
        x_out = x_attn2[:, :self.seq_len, :]  
        final_out = self.out_proj(x_out)      
        if return_weights:
            return final_out, [attn1_weights, attn2_weights]
        return final_out


def run_experiment_for_head(seed, num_heads, num_epochs=50, batch_size=32, lr=1e-3, seq_len=50, p=60,
                             input_dim=10, hidden_dim=32, target_dim=10, num_layers=2):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset = NPZDataset("train_data_50.npz")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MultiLayerAttentionModelExtended(input_dim=input_dim, hidden_dim=hidden_dim, target_dim=target_dim,
                                              seq_len=seq_len, p=p, num_heads=num_heads, num_layers=num_layers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            count += batch_x.size(0)
        epoch_loss_avg = epoch_loss / count
        train_losses.append(epoch_loss_avg)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss_avg:.4f}")

    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            count += batch_x.size(0)
    test_loss = total_loss / count
    return test_loss, train_losses


if __name__ == "__main__":
    # seeds = [1237,1238,1239,1240,1241,1242,1243]
    seeds = [1234, 1235, 1236]
    head_list = [8]
    p_list = [10,12,14,16,18,20]
    all_train_losses = {}
    file_path = "seq_10_p_experiment_results_8_small.csv"
    if not os.path.isfile(file_path):
        pd.DataFrame(columns=["seed", "p", "test_loss"]).to_csv(file_path, index=False)

    results = []
    for num_heads in head_list:
        for p in p_list:
            for seed in seeds:
                test_loss, train_losses = run_experiment_for_head(seed, num_heads, num_epochs=3, batch_size=32, lr=1e-3,
                                                                  seq_len=10, p=p, input_dim=5, hidden_dim=16,
                                                                  target_dim=5, num_layers=2)
                all_train_losses[p] = train_losses
                results.append({"seed": seed, "p": p, "test_loss": test_loss})
                print(f"Seed {seed}, p {p}: Test Loss = {test_loss:.4f}")
                df = pd.DataFrame([results[-1]])
                df.to_csv(file_path, mode='a', header=not os.path.isfile(file_path), index=False)


    plt.figure(figsize=(10, 6))
    for p, losses in all_train_losses.items():
        plt.plot(range(1, len(losses)+1), losses, marker="o", label=f"{p} p(s)")
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Training Loss (MSE)", fontsize=16)
    plt.title("Training Loss Curve for Each p", fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle="--", linewidth=0.7)
    plt.tight_layout()
    plt.savefig("training_loss_curves_2layer_p_8_10_small.pdf", bbox_inches="tight")
    plt.show()

    df = pd.read_csv(file_path)
    # df = pd.DataFrame(results)
    grouped = df.groupby("p")["test_loss"].agg(["mean", "std"]).reset_index()
    print(grouped)

    sns.set_context("notebook", font_scale=1.4, rc={"lines.linewidth": 2.5})
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 8))
    plt.plot(grouped["p"], grouped["mean"], marker="o", color="#1b7837")
    plt.fill_between(grouped["p"],
                     grouped["mean"] - grouped["std"],
                     grouped["mean"] + grouped["std"],
                     color="#1b7837", alpha=0.3)
    plt.xlabel("p", fontsize=25)
    plt.ylabel("Test MSE Loss", fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.savefig("p_experiment_results_2layer_8_10_small.pdf", bbox_inches="tight")
    plt.show()



    seeds = [1237,1238,1239,1240,1241,1242,1243]
    # head_list = [32]
    head_list = [1, 2, 4, 8, 16,32]
    all_train_losses = {}
    file_path = "seq_multihead_experiment_results_32.csv"
    if not os.path.isfile(file_path):
        pd.DataFrame(columns=["seed", "num_heads", "test_loss"]).to_csv(file_path, index=False)

    results = []
    for num_heads in head_list:
        for seed in seeds:
            test_loss, train_losses = run_experiment_for_head(seed, num_heads, num_epochs=3, batch_size=32, lr=1e-3,
                                                               seq_len=20, p=20, input_dim=5, hidden_dim=32,
                                                               target_dim=5, num_layers=2)
            all_train_losses[num_heads] = train_losses
            results.append({"seed": seed, "num_heads": num_heads, "test_loss": test_loss})
            print(f"Seed {seed}, num_heads {num_heads}: Test Loss = {test_loss:.4f}")
            df = pd.DataFrame([results[-1]])
            df.to_csv(file_path, mode='a', header=not os.path.isfile(file_path), index=False)


    plt.figure(figsize=(10, 6))
    for num_heads, losses in all_train_losses.items():
        plt.plot(range(1, len(losses)+1), losses, marker="o", label=f"{num_heads} Head(s)")
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Training Loss (MSE)", fontsize=16)
    plt.title("Training Loss Curve for Each Number of Heads", fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle="--", linewidth=0.7)
    plt.tight_layout()
    plt.savefig("training_loss_curves_2layer_32.pdf", bbox_inches="tight")
    plt.show()

    df = pd.read_csv(file_path)
    # df = pd.DataFrame(results)
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
    plt.ylabel("Test MSE Loss", fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.savefig("head_experiment_results_2layer_32.pdf", bbox_inches="tight")
    plt.show()