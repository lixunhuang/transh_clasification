import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import cv2
import numpy as np
import time

# Define Constants (match your training code)
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

# Categories (match your training code)
categories = {
    0: 'battery',
    1: 'biological',
    2: 'brown-glass',
    3: 'cardboard',
    4: 'clothes',
    5: 'green-glass',
    6: 'metal',
    7: 'paper',
    8: 'plastic',
    9: 'shoes',
    10: 'trash',
    11: 'white-glass'
}

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Model Definition (exact same as training)
class GarbageClassifier(nn.Module):
    def __init__(self, num_classes=12):
        super(GarbageClassifier, self).__init__()
        # Load pretrained MobileNetV2 without top
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet.classifier = nn.Identity()  # Remove the classifier

        # Freeze the backbone (as in training)
        for param in self.mobilenet.parameters():
            param.requires_grad = False

        # Global Average Pooling + Final Dense
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.mobilenet.last_channel, num_classes)  # 1280 for MobileNetV2

    def forward(self, x):
        x = self.mobilenet.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Load the trained model
model = GarbageClassifier(num_classes=len(categories)).to(device)
model.load_state_dict(torch.load("model12.pth", map_location=device))
model.eval()  # Set to evaluation mode
print('Model loaded successfully!')

# Transforms (use val_test_transform from training)
val_test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Camera setup
cap = cv2.VideoCapture(0)  # Use default camera (0). Change to 1 or path if external.
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Starting real-time inference. Press 'q' to quit.")

# FPS calculation
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Flip frame horizontally for mirror effect (optional)
    frame = cv2.flip(frame, 1)

    # Prepare image for model: Convert BGR to RGB, then PIL, then transform
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Preprocess
    input_tensor = val_test_transform(pil_image).unsqueeze(0).to(device)  # Add batch dim

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()
        confidence = torch.softmax(outputs, dim=1)[0][predicted_class].item() * 100

    # Get class name
    class_name = categories[predicted_class]

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display on frame
    cv2.putText(frame, f'Class: {class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Confidence: {confidence:.1f}%', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow('Garbage Classifier - Real-time Inference', frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Inference stopped.")