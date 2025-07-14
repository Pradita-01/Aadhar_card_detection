import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image  # To load a single image for prediction


# Paths and settings
train_dir = "dataset/train"
val_dir = "dataset/validation"
batch_size = 4
img_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")


# Data transforms
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Datasets and loaders
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class_names = train_dataset.classes
print(f"Classes: {class_names}")


# Load pretrained CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Freeze CLIP weights
for param in clip_model.parameters():
    param.requires_grad = False

embed_dim = clip_model.config.projection_dim

# Define custom classifier head
classifier = nn.Sequential(
    nn.Linear(embed_dim, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, len(class_names))  # Automatically handle number of classes
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=1e-4)


# Training loop
epochs = 10
for epoch in range(epochs):
    classifier.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in tqdm(train_loader, desc=f" Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)

        # Extract image features with CLIP
        with torch.no_grad():
            image_features = clip_model.get_image_features(pixel_values=images)

        # Forward pass through classifier
        outputs = classifier(image_features)
        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.2f}%")


# Validation accuracy
classifier.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)

        image_features = clip_model.get_image_features(pixel_values=images)
        outputs = classifier(image_features)
        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

val_acc = 100 * correct / total
print(f" Validation Accuracy: {val_acc:.2f}%")


# 🔥 Function: Predict a single image (real/fake)
def predict_image(image_path):
    classifier.eval()
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f" Could not open image: {e}")
        return

    transformed = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        image_features = clip_model.get_image_features(pixel_values=transformed)
        outputs = classifier(image_features)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
        print(f"The image is predicted as: **{predicted_class}**")


# 🔥 Function: Show predictions for validation data
def show_predictions(loader, num_images=6):
    classifier.eval()
    images_shown = 0
    plt.figure(figsize=(15, 5))
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            image_features = clip_model.get_image_features(pixel_values=images)
            outputs = classifier(image_features)
            _, predicted = torch.max(outputs, 1)

            for i in range(images.size(0)):
                if images_shown >= num_images:
                    break
                img = images[i].permute(1, 2, 0).cpu().numpy()
                pred_label = class_names[predicted[i]]
                plt.subplot(1, num_images, images_shown + 1)
                plt.imshow((img * 0.5) + 0.5)  # Unnormalize
                plt.title(f"Pred: {pred_label}")
                plt.axis('off')
                images_shown += 1
    plt.show()


# Example usage: Predict a single image
test_image_path = r"test_images/sample.png"  # Replace with your own image path
predict_image(test_image_path)

# 🚀 Example usage: Show predictions for validation set
show_predictions(val_loader)
