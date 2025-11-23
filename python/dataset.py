import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import glob
import pandas as pd

class CSVImageDataset(Dataset):
    """Dataset untuk gambar dengan label dari CSV file"""
    def __init__(self, csv_path, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # Baca CSV
        self.df = pd.read_csv(csv_path)
        
        # Dapatkan unique classes dan buat mapping
        self.classes = sorted(self.df['label'].unique().tolist())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['filename']
        label_name = row['label']
        
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.class_to_idx[label_name]
        return image, label

class FlatImageDataset(Dataset):
    """Dataset untuk gambar yang berada dalam satu folder tanpa subfolder kelas"""
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.[jJ][pP]*[gG]")))
        # Untuk single class classification, semua gambar adalah kelas 0
        self.classes = ["images"]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Semua gambar dikategorikan sebagai kelas 0 (single class)
        label = 0
        return image, label

def get_dataloaders(data_dir, batch_size=32, img_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(data_dir, "train")
    train_csv = os.path.join(data_dir, "train.csv")
    test_csv = os.path.join(data_dir, "test.csv")
    test_dir = os.path.join(data_dir, "test") if os.path.exists(os.path.join(data_dir, "test")) else train_dir

    # Cek apakah ada CSV file untuk labels
    if os.path.exists(train_csv):
        # Gunakan CSV-based dataset
        train_dataset = CSVImageDataset(train_csv, train_dir, transform=train_transform)
        if os.path.exists(test_csv):
            test_dataset = CSVImageDataset(test_csv, test_dir, transform=test_transform)
        else:
            # Gunakan train dataset untuk test jika test.csv tidak ada
            test_dataset = CSVImageDataset(train_csv, train_dir, transform=test_transform)
    else:
        # Cek apakah menggunakan struktur ImageFolder atau flat structure
        subdirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        
        if len(subdirs) > 0:
            # Struktur dengan subfolder kelas
            train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
            test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
        else:
            # Struktur flat (semua gambar dalam satu folder)
            train_dataset = FlatImageDataset(train_dir, transform=train_transform)
            test_dataset = FlatImageDataset(test_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, train_dataset.classes
