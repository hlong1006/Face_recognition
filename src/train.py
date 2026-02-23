import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .model import CNN 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
NUM_CLASSES = 64

def run_train(data_root='data'):
    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),    
        transforms.ToTensor(),           
       
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

 
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val') 
    if not os.path.exists(val_dir):
        val_dir = os.path.join(data_root, 'test')

   
    if not os.path.exists(train_dir):
        print(f"Error: Không tìm thấy thư mục '{train_dir}")
        return

    train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    has_val = os.path.exists(val_dir)
    if has_val:
        val_dataset = datasets.ImageFolder(root=val_dir, transform=data_transform)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    else:
        print(f"Warning: Không tìm thấy thư mục '{val_dir}'. Training only, no validation.")
        val_loader = []

    print(f"Tìm thấy {len(train_dataset)} ảnh train thuộc {len(train_dataset.classes)} lớp: {train_dataset.classes}")

   
    model = CNN(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_acc = 0.0
    best_train_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 50 == 0: 
                current_acc = 100.0 * correct_train / total_train if total_train else 0.0
                print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{batch_idx+1}/{len(train_loader)}] - "
                      f"Loss: {running_loss/(batch_idx+1):.4f}, Acc: {current_acc:.2f}%")

        train_acc = 100.0 * correct_train / total_train if total_train else 0.0
        avg_train_loss = running_loss / len(train_loader) if len(train_loader) else 0.0
      
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        if has_val:
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_acc = 100.0 * correct_val / total_val if total_val else 0.0
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) else 0.0

        log = f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%"
        if has_val:
            log += f" | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        print(log)
        to_save = (has_val and val_acc > best_val_acc) or (not has_val and train_acc > best_train_acc)
        if has_val and val_acc > best_val_acc:
            best_val_acc = val_acc
        if not has_val and train_acc > best_train_acc:
            best_train_acc = train_acc

        if to_save:
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_model.pth')
            print("--> Saved best model!")

    print("Training finished!")

