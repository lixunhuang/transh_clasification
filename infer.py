import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import cv2
import numpy as np
import os

# ================= 配置区域 =================
# 1. 这里填你的模型路径
MODEL_PATH = "model12.pth"

# 2. 这里填你想测试的文件夹路径 (或者单张图片路径)
TEST_DIR = "garbage_classification/paper/"  # 请确保这个文件夹存在，并且里面有几张 jpg/png 图片

# ===========================================

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


# ------------------------------------------------------
# 1. 模型定义 (直接复制你的 inference.py，一个字不改)
# ------------------------------------------------------
class GarbageClassifier(nn.Module):
    def __init__(self, num_classes=12):
        super(GarbageClassifier, self).__init__()
        # Load pretrained MobileNetV2 without top
        self.mobilenet = models.mobilenet_v2(pretrained=False)  # 推理时不需要下载预训练权重，因为我们会加载你训练好的 model12.pth
        self.mobilenet.classifier = nn.Identity()  # Remove the classifier

        # Global Average Pooling + Final Dense
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.mobilenet.last_channel, num_classes)  # 1280 for MobileNetV2

    def forward(self, x):
        x = self.mobilenet.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ------------------------------------------------------
# 2. 类别和预处理 (和你训练时保持一致)
# ------------------------------------------------------
categories = categories = {
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

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ------------------------------------------------------
# 3. 加载权重 (关键步骤)
# ------------------------------------------------------
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到文件 {MODEL_PATH}")
        exit()

    print(f"正在加载 {MODEL_PATH} ...")
    model = GarbageClassifier(num_classes=len(categories)).to(device)

    try:
        # 加载你辛苦训练出来的权重 (state_dict)
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()  # 开启评估模式 (固定 BatchNorm 等)
        print("✅ 模型加载成功！权重已注入！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        exit()

    return model


# ------------------------------------------------------
# 4. 预测逻辑
# ------------------------------------------------------
def predict_image(model, image_path):
    # 读取图片
    try:
        pil_image = Image.open(image_path).convert('RGB')
    except:
        print(f"无法读取图片: {image_path}")
        return

    # 预处理
    input_tensor = val_test_transform(pil_image).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        # 获取第一名
        top_prob, top_catid = torch.max(probabilities, 1)

        # 获取数据
        class_id = top_catid.item()
        class_name = categories[class_id]
        confidence = top_prob.item() * 100

    # --- 显示结果 (OpenCV) ---
    print(f"图片: {os.path.basename(image_path)} -> 预测: {class_name} ({confidence:.1f}%)")

    # 转成 OpenCV 格式显示
    opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # 调整显示大小，太大的图缩小一点，太小的图放大一点
    h, w = opencv_image.shape[:2]
    if w > 800:
        scale = 800 / w
        opencv_image = cv2.resize(opencv_image, (0, 0), fx=scale, fy=scale)

    # 绘制文字
    label_text = f"{class_name}: {confidence:.1f}%"
    color = (0, 255, 0)  # 绿色
    if class_name == "metal": color = (0, 0, 255)  # 如果是 Metal 显示红色，方便你看是不是又错了

    cv2.putText(opencv_image, label_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)  # 黑边
    cv2.putText(opencv_image, label_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)  # 彩字

    cv2.imshow("Prediction", opencv_image)
    print("按任意键查看下一张 (按 'q' 退出)...")
    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        exit()


# ------------------------------------------------------
# 主程序
# ------------------------------------------------------
if __name__ == "__main__":
    # 1. 准备模型
    model = load_trained_model()

    # 2. 准备图片列表
    if os.path.isfile(TEST_DIR):  # 如果配置的是单张图片
        image_files = [TEST_DIR]
    elif os.path.isdir(TEST_DIR):  # 如果配置的是文件夹
        image_files = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()
    else:
        print(f"❌ 找不到测试路径: {TEST_DIR}")
        exit()

    if not image_files:
        print("❌ 文件夹里没有图片！")
        exit()

    # 3. 循环预测
    for img_path in image_files:
        predict_image(model, img_path)

    cv2.destroyAllWindows()