import torch
import numpy as np

def generate_data(num_samples, seq_length, d, W1, b1, W2, b2):

    # uniformly in [0, 1]
    X = torch.rand(num_samples, seq_length, d)

    # shape (num_samples, seq_length * d)
    X_flat = X.view(num_samples, -1)


    hidden = torch.relu(torch.matmul(X_flat, W1.t()) + b1)


    Y_flat = torch.matmul(hidden, W2.t()) + b2

    Y = Y_flat.view(num_samples, seq_length, d)

    return X, Y

if __name__ == "__main__":

    torch.manual_seed(42)


    num_train = 50000
    num_test = 2000
    seq_length = 20  
    d = 5          
    input_dim = seq_length * d


    hidden_dim = 10


    W1 = torch.randn(hidden_dim, input_dim)
    b1 = torch.randn(hidden_dim)
    W2 = torch.randn(input_dim, hidden_dim)
    b2 = torch.randn(input_dim)


    X_train, Y_train = generate_data(num_train, seq_length, d, W1, b1, W2, b2)

    X_test, Y_test = generate_data(num_test, seq_length, d, W1, b1, W2, b2)


    np.savez("train_data.npz", X_train=X_train.numpy(), Y_train=Y_train.numpy())
    np.savez("test_data.npz", X_test=X_test.numpy(), Y_test=Y_test.numpy())
