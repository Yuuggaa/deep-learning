import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
import json

from dataset import get_dataloaders
from model_swin import create_swin
from model_vit import create_vit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "dataset"
BATCH_SIZE = 32

def measure_inference_time(model, test_loader, device):
    model.eval()
    total_time = 0.0
    num_images = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            # Warm up
            if num_images == 0:
                _ = model(images)
            
            start_time = time.time()
            outputs = model(images)
            
            # Sync for accurate timing on GPU
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            batch_time = end_time - start_time
            total_time += batch_time
            num_images += images.size(0)
    
    avg_time_per_image = (total_time / num_images) * 1000  # Convert to ms
    throughput = num_images / total_time  # Images per second
    
    return avg_time_per_image, total_time, throughput, num_images

def evaluate_model(model, test_loader, classes, model_name):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate metrics
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate per-class and average metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=range(len(classes))
    )
    
    # Calculate weighted averages
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    # Measure inference time
    avg_time_ms, total_time, throughput, num_images = measure_inference_time(model, test_loader, device)
    
    # Get hardware info
    if device.type == 'cuda':
        hardware_info = f"GPU: {torch.cuda.get_device_name(0)}"
    else:
        hardware_info = "CPU"
    
    # Print detailed metrics
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS: {model_name}")
    print("="*60)
    print(f"\nA. MODEL PARAMETERS")
    print(f"   (Already printed during training)")
    
    print(f"\nB. PERFORMANCE METRICS")
    print(f"   Overall Accuracy: {report['accuracy']*100:.2f}%")
    print(f"\n   Per-Class Metrics:")
    for i, class_name in enumerate(classes):
        print(f"   {class_name}:")
        print(f"      Precision: {precision[i]*100:.2f}%")
        print(f"      Recall: {recall[i]*100:.2f}%")
        print(f"      F1-Score: {f1[i]*100:.2f}%")
        print(f"      Support: {support[i]}")
    
    print(f"\n   Weighted Average:")
    print(f"      Precision: {precision_avg*100:.2f}%")
    print(f"      Recall: {recall_avg*100:.2f}%")
    print(f"      F1-Score: {f1_avg*100:.2f}%")
    
    print(f"\nC. INFERENCE TIME")
    print(f"   Average time per image: {avg_time_ms:.2f} ms")
    print(f"   Total time for {num_images} images: {total_time:.2f} seconds")
    print(f"   Throughput: {throughput:.2f} images/second")
    print(f"   Hardware: {hardware_info}")
    print("="*60 + "\n")

    return report, cm, {
        'avg_time_ms': avg_time_ms,
        'total_time': total_time,
        'throughput': throughput,
        'num_images': num_images,
        'hardware': hardware_info,
        'per_class_metrics': {
            classes[i]: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            } for i in range(len(classes))
        },
        'weighted_avg': {
            'precision': float(precision_avg),
            'recall': float(recall_avg),
            'f1_score': float(f1_avg)
        }
    }


def main():
    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    train_loader, test_loader, classes = get_dataloaders(DATA_DIR, BATCH_SIZE)

    # ===== Evaluate Swin =====
    print("Loading Swin Transformer model...")
    swin = create_swin(len(classes)).to(device)
    swin.load_state_dict(torch.load("models/swin_model.pth"))
    rep_swin, cm_swin, inference_swin = evaluate_model(swin, test_loader, classes, "Swin Transformer")
    
    # Save metrics
    pd.DataFrame(rep_swin).to_csv("outputs/results/swin_metrics.csv")
    
    # Save detailed metrics
    with open("outputs/results/swin_detailed_metrics.json", "w") as f:
        json.dump({
            'classification_report': rep_swin,
            'inference_metrics': inference_swin
        }, f, indent=2)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_swin, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Swin Transformer Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.tight_layout()
    plt.savefig("outputs/figures/swin_cm.png", dpi=200)
    plt.close()

    print("Swin Transformer evaluation saved!")

    # ===== Evaluate ViT =====
    print("\nLoading Vision Transformer (ViT) model...")
    vit = create_vit(len(classes)).to(device)
    vit.load_state_dict(torch.load("models/vit_model.pth"))
    rep_vit, cm_vit, inference_vit = evaluate_model(vit, test_loader, classes, "Vision Transformer (ViT)")
    
    # Save metrics
    pd.DataFrame(rep_vit).to_csv("outputs/results/vit_metrics.csv")
    
    # Save detailed metrics
    with open("outputs/results/vit_detailed_metrics.json", "w") as f:
        json.dump({
            'classification_report': rep_vit,
            'inference_metrics': inference_vit
        }, f, indent=2)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_vit, annot=True, fmt="d", cmap="Greens", xticklabels=classes, yticklabels=classes)
    plt.title("Vision Transformer (ViT) Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.tight_layout()
    plt.savefig("outputs/figures/vit_cm.png", dpi=200)
    plt.close()

    print("Vision Transformer (ViT) evaluation saved!")
    
    print("\n" + "="*60)
    print("All evaluations completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
