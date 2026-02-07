"""
Minimal PyTorch MNIST classifier with comprehensive MLflow logging.

This script demonstrates MLflow's experiment tracking capabilities:
- Automatic logging via mlflow.pytorch.autolog()
- Manual parameter logging
- Metric logging at each step/epoch
- Model artifact logging
- Custom artifact logging (plots, etc.)
"""

import argparse
import os

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTNet(nn.Module):
    """Simple feedforward neural network for MNIST classification."""

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def get_data_loaders(batch_size: int):
    """Create train and test data loaders for MNIST."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, device, epoch, log_interval=100):
    """Train for one epoch and log metrics."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)

        # Log batch-level metrics
        global_step = epoch * len(train_loader) + batch_idx
        if batch_idx % log_interval == 0:
            mlflow.log_metrics({
                "train_loss_step": loss.item(),
                "train_accuracy_step": correct / total,
            }, step=global_step)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy


def save_training_curve(train_losses, test_losses, train_accs, test_accs):
    """Save training curves as artifact."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        epochs = range(1, len(train_losses) + 1)

        ax1.plot(epochs, train_losses, "b-", label="Train Loss")
        ax1.plot(epochs, test_losses, "r-", label="Test Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Test Loss")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs, train_accs, "b-", label="Train Accuracy")
        ax2.plot(epochs, test_accs, "r-", label="Test Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training and Test Accuracy")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        # Save and log as artifact
        os.makedirs("artifacts", exist_ok=True)
        fig_path = "artifacts/training_curves.png"
        plt.savefig(fig_path, dpi=100)
        plt.close()
        mlflow.log_artifact(fig_path)
        print(f"Saved training curves to {fig_path}")
    except ImportError:
        print("matplotlib not available, skipping training curve plot")


def main():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example with MLflow")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128)
    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Enable MLflow autolog for PyTorch
    # This automatically logs parameters, metrics, and model
    mlflow.pytorch.autolog(log_models=False)  # We'll log model manually for demo

    with mlflow.start_run(run_id=os.environ.get("MLFLOW_RUN_ID")):
        # Log additional parameters manually (autolog captures some automatically)
        mlflow.log_params({
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "hidden_size": args.hidden_size,
            "device": str(device),
            "optimizer": "SGD",
            "loss_function": "NLLLoss",
        })

        # Log system info as tags
        mlflow.set_tags({
            "framework": "pytorch",
            "task": "image_classification",
            "dataset": "MNIST",
        })

        # Initialize data loaders and model
        train_loader, test_loader = get_data_loaders(args.batch_size)
        model = MNISTNet(hidden_size=args.hidden_size).to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

        # Log model architecture as text artifact
        mlflow.log_text(str(model), "model_architecture.txt")

        # Training loop
        train_losses, test_losses = [], []
        train_accs, test_accs = [], []

        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")

            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, device, epoch
            )
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # Evaluate
            test_loss, test_acc = evaluate(model, test_loader, device)
            test_losses.append(test_loss)
            test_accs.append(test_acc)

            # Log epoch-level metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
            }, step=epoch)

            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        # Log final metrics
        mlflow.log_metrics({
            "final_test_loss": test_losses[-1],
            "final_test_accuracy": test_accs[-1],
            "best_test_accuracy": max(test_accs),
        })

        # Save training curves as artifact
        save_training_curve(train_losses, test_losses, train_accs, test_accs)

        # Log model with signature
        from mlflow.models import infer_signature
        sample_input = torch.randn(1, 1, 28, 28).to(device)
        sample_output = model(sample_input).detach().cpu().numpy()
        signature = infer_signature(
            sample_input.cpu().numpy(),
            sample_output
        )
        mlflow.pytorch.log_model(
            model,
            name="mnist_model",
            signature=signature,
        )

        print(f"\nTraining complete!")
        print(f"Best test accuracy: {max(test_accs):.4f}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
