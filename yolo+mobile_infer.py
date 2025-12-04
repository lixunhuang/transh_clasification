import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import time
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO  # pip install ultralytics

# -----------------------
# Constants & Categories
# -----------------------
IMAGE_SIZE = (224, 224)
categories = {0: 'paper', 1: 'cardboard', 2: 'plastic', 3: 'metal', 4: 'trash', 5: 'battery',
              6: 'shoes', 7: 'clothes', 8: 'green-glass', 9: 'brown-glass', 10: 'white-glass',
              11: 'biological'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------
# MobileNet Classifier
# -----------------------
class GarbageClassifier(nn.Module):
    def __init__(self, num_classes=12):
        super(GarbageClassifier, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet.classifier = nn.Identity()
        for param in self.mobilenet.parameters():
            param.requires_grad = False
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.mobilenet.last_channel, num_classes)

    def forward(self, x):
        x = self.mobilenet.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Load trained MobileNet
model = GarbageClassifier(len(categories)).to(device)
model.load_state_dict(torch.load("model12.pth", map_location=device))
model.eval()
print("MobileNet model loaded.")

val_test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# -----------------------
# Load YOLOv8 pre-trained model
# -----------------------
yolo_model = YOLO('yolov8n.pt')  # yolov8n pre-trained on COCO
print("YOLOv8 model loaded.")

# -----------------------
# Camera setup
# -----------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # optional mirror

    # YOLOv8 detection
    results = yolo_model(frame)[0]  # process current frame
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        if conf < 0.3:  # filter low-confidence detections
            continue

        # Crop detected object
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        # Prepare image for MobileNet
        rgb_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_cropped)
        input_tensor = val_test_transform(pil_img).unsqueeze(0).to(device)

        # MobileNet classification
        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred = torch.max(outputs, 1)
            pred_class = pred.item()
            confidence = torch.softmax(outputs, dim=1)[0][pred_class].item() * 100
            class_name = categories[pred_class]

        # Draw bbox and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f'{class_name} {confidence:.1f}%', (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {fps:.1f}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

    # Show frame
    cv2.imshow('Garbage Detection & Classification', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
