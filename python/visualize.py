import matplotlib.pyplot as plt
import json
import os

def plot_learning_curve(log_file, title):
    if not os.path.exists(log_file):
        print(f"Warning: {log_file} not found. Skipping {title}.")
        return
        
    with open(log_file, "r") as f:
        log = json.load(f)

    # Support both old and new format
    if "train_loss" in log:
        epochs = list(range(1, len(log["train_loss"]) + 1))
        train_loss = log["train_loss"]
        train_acc = log["train_accuracy"]
        val_loss = log["val_loss"]
        val_acc = log["val_accuracy"]
    else:
        # Old format compatibility
        epochs = list(range(1, len(log["loss"]) + 1))
        train_loss = log["loss"]
        train_acc = log.get("accuracy", [])
        val_loss = []
        val_acc = []
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Loss
    ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='Train Loss')
    if val_loss:
        ax1.plot(epochs, val_loss, 'r-', linewidth=2, label='Val Loss')
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title(f"{title} - Loss", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Accuracy
    if train_acc:
        ax2.plot(epochs, train_acc, 'b-', linewidth=2, label='Train Accuracy')
    if val_acc:
        ax2.plot(epochs, val_acc, 'r-', linewidth=2, label='Val Accuracy')
    if train_acc or val_acc:
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Accuracy (%)", fontsize=12)
        ax2.set_title(f"{title} - Accuracy", fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"outputs/figures/{title.replace(' ', '_').lower()}.png", dpi=200)
    plt.close()
    print(f"{title} saved!")


def main():
    os.makedirs("outputs/figures", exist_ok=True)

    plot_learning_curve("outputs/results/swin_log.json", "Swin Training Curve")
    plot_learning_curve("outputs/results/vit_log.json", "ViT Training Curve")

    print("\nTraining curves generation complete!")
    print("Note: Run training scripts first to generate log files if they don't exist.")


if __name__ == "__main__":
    main()
