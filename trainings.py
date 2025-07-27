import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Paths and settings
train_dir = "dataset/train"
val_dir = "dataset/validation"
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

# Load pretrained CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Freeze all CLIP layers first (then we can unfreeze later for fine-tuning)
for param in clip_model.parameters():
    param.requires_grad = False

# Custom classifier head
embed_dim = clip_model.config.projection_dim
classifier = nn.Sequential(
    nn.Linear(embed_dim, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2)  # 2 classes: real, fake
).to(device)

# Dataset using CLIP preprocessing
class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, processor):
        self.dataset = datasets.ImageFolder(root=root_dir)
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return image, label

train_dataset = CLIPDataset(train_dir, clip_processor)
val_dataset = CLIPDataset(val_dir, clip_processor)

# Fix class imbalance
labels = [label for _, label in train_dataset.dataset.samples]
class_counts = torch.bincount(torch.tensor(labels))
class_weights = 1. / class_counts.float()
sample_weights = [class_weights[label] for label in labels]
sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f" Class weights: {class_weights}")

# Focal Loss to handle imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

criterion = FocalLoss()
optimizer = optim.AdamW(classifier.parameters(), lr=2e-4)

# Training loop
epochs = 15
best_val_acc = 0

for epoch in range(epochs):
    classifier.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():  # Freeze CLIP encoder during this phase
            image_features = clip_model.get_image_features(pixel_values=images)

        outputs = classifier(image_features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    print(f" Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")

    # Validation
    classifier.eval()
    val_correct, val_total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            image_features = clip_model.get_image_features(pixel_values=images)
            outputs = classifier(image_features)
            _, predicted = torch.max(outputs, 1)

            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = 100 * val_correct / val_total
    print(f" Validation Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'clip_model_state_dict': clip_model.state_dict(),
            'classifier_state_dict': classifier.state_dict()
        }, "best_model.pth")
        print(" Best model saved!")

# Classification report
print("\n Final Classification Report:")
print(classification_report(all_labels, all_preds, target_names=train_dataset.dataset.classes))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=train_dataset.dataset.classes,
            yticklabels=train_dataset.dataset.classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
