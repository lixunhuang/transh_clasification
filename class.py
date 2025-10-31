import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

print('setup successful!')
# %% md
# Define Constants
# %%
# Increasing the image size didn't result in increasing the training accuracy
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

# Path where our data is located
base_path = "garbage_classification/"

# Dictionary to save our 12 classes
categories = {0: 'paper', 1: 'cardboard', 2: 'plastic', 3: 'metal', 4: 'trash', 5: 'battery',
              6: 'shoes', 7: 'clothes', 8: 'green-glass', 9: 'brown-glass', 10: 'white-glass',
              11: 'biological'}

print('defining constants successful!')
# %% md
# Create DataFrame
# %% md

# %%
# Add class name prefix to filename. So for example "/paper104.jpg" become "paper/paper104.jpg"
def add_class_name_prefix(df, col_name):
    df[col_name] = df[col_name].apply(lambda x: x[:re.search("\d", x).start()] + '/' + x)
    return df


# list conatining all the filenames in the dataset
filenames_list = []
# list to store the corresponding category, note that each folder of the dataset has one class of data
categories_list = []

for category in categories:
    filenames = os.listdir(base_path + categories[category])

    filenames_list = filenames_list + filenames
    categories_list = categories_list + [category] * len(filenames)

df = pd.DataFrame({
    'filename': filenames_list,
    'category': categories_list
})

df = add_class_name_prefix(df, 'filename')

# Shuffle the dataframe
df = df.sample(frac=1).reset_index(drop=True)

print('number of elements = ', len(df))
# %%
df.head()
# %%
# see sample image, you can run the same cell again to get a different image
random_row = random.randint(0, len(df) - 1)
sample = df.iloc[random_row]
randomimage = Image.open(base_path + sample['filename'])
print(sample['filename'])
plt.imshow(randomimage)
plt.show()
# %% md
# Viusalize the Categories Distribution
# %%
df_visualization = df.copy()
# Change the catgegories from numbers to names
df_visualization['category'] = df_visualization['category'].apply(lambda x: categories[x])

df_visualization['category'].value_counts().plot.bar(x='count', y='category')

plt.xlabel("Garbage Classes", labelpad=14)
plt.ylabel("Images Count", labelpad=14)
plt.title("Count of images per class", y=1.02);
plt.show()
# %% md
# Create the model
# %% md


# %%
# Custom MobileNetV2 preprocessing (similar to Keras)
def mobilenetv2_preprocessing(img):
    # Normalize to [-1, 1] as per MobileNetV2
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (img - mean) / std
    return img


class GarbageClassifier(nn.Module):
    def __init__(self, num_classes=12):
        super(GarbageClassifier, self).__init__()
        # Load pretrained MobileNetV2 without top
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet.classifier = nn.Identity()  # Remove the classifier
        self.mobilenet.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1,
                                                  bias=False)  # Adjust if needed, but pretrained should work

        # Freeze the backbone
        for param in self.mobilenet.parameters():
            param.requires_grad = False

        # Global Average Pooling + Final Dense
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.mobilenet.last_channel, num_classes)  # last_channel is 1280 for MobileNetV2

    def forward(self, x):
        x = self.mobilenet.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


model = GarbageClassifier(num_classes=len(categories)).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # Only optimize the final layer

print(model)


class EarlyStopping:
    def __init__(self, patience=2, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_accuracy, model):
        if self.best_loss is None:
            self.best_loss = val_accuracy
            self.save_checkpoint(model)
        elif val_accuracy > self.best_loss + self.min_delta:
            self.best_loss = val_accuracy
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


early_stop = EarlyStopping(patience=2, min_delta=0.001)

print('call back defined!')
# %% md
# Split the Data Set
# %% md

# %%
# Change the categories from numbers to names
df["category"] = df["category"].replace(categories)

# We first split the data into two sets and then split the validate_df to two sets
train_df, validate_df = train_test_split(df, test_size=0.2, random_state=42)
validate_df, test_df = train_test_split(validate_df, test_size=0.5, random_state=42)

train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]

print('train size = ', total_train, 'validate size = ', total_validate, 'test size = ', test_df.shape[0])


# %% md
# Custom Dataset Class
# %%
class GarbageDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, is_train=False):
        self.labels = pd.Categorical(dataframe['category']).codes  # 保持原样
        self.file_names = dataframe['filename'].values
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.class_to_idx = {cat: idx for idx, cat in enumerate(sorted(set(dataframe['category'])))}

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.file_names[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]  # int8 或 int32

        if self.transform:
            image = self.transform(image)

        # 修复：强制转换为 torch.long
        label = torch.tensor(label, dtype=torch.long)

        return image, label


# Transforms
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    # Augmentation (uncomment if needed)
    # transforms.RandomRotation(30),
    # transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # MobileNetV2 normalization
])

val_test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# DataLoaders
batch_size = 64

train_dataset = GarbageDataset(train_df, base_path, transform=train_transform, is_train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

val_dataset = GarbageDataset(validate_df, base_path, transform=val_test_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

test_dataset = GarbageDataset(test_df, base_path, transform=val_test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

print('Data loaders created!')
# %% md
# Train the model
# %% md

EPOCHS = 20
history = {'loss': [], 'categorical_accuracy': [], 'val_loss': [], 'val_categorical_accuracy': []}

for epoch in range(EPOCHS):
    # Training
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train
    history['loss'].append(train_loss)
    history['categorical_accuracy'].append(train_acc)

    # Validation
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss = running_val_loss / len(val_loader)
    val_acc = 100 * correct_val / total_val
    history['val_loss'].append(val_loss)
    history['val_categorical_accuracy'].append(val_acc)

    print(
        f'Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    # Early stopping check (monitor val_acc)
    # if early_stop(val_acc / 100, model):  # Pass accuracy as fraction
    #     print(f'Early stopping at epoch {epoch + 1}')
    #     break

print('Training complete!')
# %%
# Save model
torch.save(model.state_dict(), "model12.pth")
print('Model saved as model12.pth')
# %% md
# Visualize the training process

# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(history['loss'], color='b', label="Training loss")
ax1.plot(history['val_loss'], color='r', label="validation loss")
ax1.set_yticks(np.arange(0, 0.7, 0.1))
ax1.legend()
ax1.set_title('Model Loss')

ax2.plot(history['categorical_accuracy'], color='b', label="Training accuracy")
ax2.plot(history['val_categorical_accuracy'], color='r', label="Validation accuracy")
ax2.legend()
ax2.set_title('Model Accuracy')

plt.tight_layout()
plt.show()
# %% md
# Evaluate the test
# %% md

model.eval()
correct_test = 0
total_test = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct_test / total_test
print('Accuracy on test set = ', round(accuracy, 2), '%')
# %%
# Get class names for predictions
gen_label_map = {i: cat for i, cat in enumerate(sorted(set(test_df['category'])))}
print(gen_label_map)

# Convert indices to names
preds = [gen_label_map[p] for p in all_preds]
labels = [gen_label_map[l] for l in all_labels]

print(classification_report(labels, preds))