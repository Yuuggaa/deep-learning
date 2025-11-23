import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import os
import json

from dataset import get_dataloaders
from model_mobilevit import create_mobilevit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

DATA_DIR = "dataset"
EPOCHS = 10
LR = 1e-4
BATCH_SIZE = 32

def train():
    train_loader, test_loader, classes = get_dataloaders(DATA_DIR, BATCH_SIZE)
    num_classes = len(classes)

    model = create_mobilevit(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_running_loss / len(test_loader)
        val_acc = 100 * val_correct / val_total
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)
    
    torch.save(model.state_dict(), "models/mobilevit_model.pth")
    print("Model saved to models/mobilevit_model.pth")
    
    with open("outputs/results/mobilevit_log.json", "w") as f:
        json.dump({
            "train_loss": train_loss_history, 
            "train_accuracy": train_acc_history,
            "val_loss": val_loss_history,
            "val_accuracy": val_acc_history
        }, f)
    print("Training log saved to outputs/results/mobilevit_log.json")

if __name__ == "__main__":
    train()
