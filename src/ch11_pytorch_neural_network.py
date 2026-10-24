"""
Chapter 11: Introduction to Deep Learning & Neural Networks in PyTorch
Modern Machine Learning & Data Science Engineering
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class BinaryClassifierMLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        return self.net(x)

def run_pytorch_training():
    print("--- PyTorch Deep Neural Network Training ---")
    torch.manual_seed(42)
    X = torch.randn(1000, 8)
    y = torch.randint(0, 2, (1000, 1)).float()

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = BinaryClassifierMLP(in_features=8)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)

    model.train()
    for epoch in range(5):
        total_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_X.size(0)
            
        print(f"Epoch {epoch+1:02d} / 05 | Loss: {total_loss / 1000:.4f}")

    model.eval()
    with torch.no_grad():
        test_sample = torch.randn(1, 8)
        prob = torch.sigmoid(model(test_sample)).item()
        print(f"\nInference on new sample: Predicted Probability = {prob:.2%}")

if __name__ == "__main__":
    run_pytorch_training()
