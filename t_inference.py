import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.quantization import QuantStub, DeQuantStub, QConfig, HistogramObserver, PerChannelMinMaxObserver
from PIL import Image
import cv2
import numpy as np
import os

# ================= 配置 =================
# 必须和你训练时保存的文件名一致
MODEL_PATH = "model_advanced_qat.pth"
IMAGE_SIZE = 224

# 字母序字典 (必须一致)
categories = {
    0: 'battery', 1: 'biological', 2: 'brown-glass', 3: 'cardboard',
    4: 'clothes', 5: 'green-glass', 6: 'metal', 7: 'paper',
    8: 'plastic', 9: 'shoes', 10: 'trash', 11: 'white-glass'
}


# ================= 1. 模型定义 (必须完全复刻训练代码) =================
class QATInvertedResidual(nn.Module):
    def __init__(self, original_block):
        super().__init__()
        self.conv = original_block.conv
        self.use_res_connect = original_block.use_res_connect
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)


def replace_with_qat_blocks(model):
    for name, child in model.named_children():
        if type(child).__name__ == 'InvertedResidual':
            setattr(model, name, QATInvertedResidual(child))
        else:
            replace_with_qat_blocks(child)


class QuantizableGarbageClassifier(nn.Module):
    def __init__(self, num_classes=12):
        super(QuantizableGarbageClassifier, self).__init__()
        self.quant = QuantStub()
        self.mobilenet = models.mobilenet_v2(pretrained=False)
        self.mobilenet.classifier = nn.Identity()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.mobilenet.last_channel, num_classes)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.mobilenet.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dequant(x)
        return x


# ================= 2. 加载并转换模型 =================
def load_and_convert_model():
    print(f"正在加载 {MODEL_PATH} ...")

    # A. 初始化空模型
    # 注意：量化模型推理通常在 CPU 上进行 (PyTorch 对 CUDA INT8 支持有限)
    device = torch.device("cpu")
    model = QuantizableGarbageClassifier(num_classes=len(categories)).to(device)

    # B. 替换结构
    replace_with_qat_blocks(model.mobilenet)

    # C. 配置量化规则 (必须和训练时一模一样，否则 key 对不上)
    model.qconfig = QConfig(
        activation=HistogramObserver.with_args(reduce_range=False),
        weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
    )

    # D. 插入观察者 (Prepare)
    torch.quantization.prepare_qat(model, inplace=True)

    # E. 加载权重
    if not os.path.exists(MODEL_PATH):
        print("❌ 找不到模型文件！请检查路径。")
        exit()

    try:
        # 加载 FP32 (带伪量化参数) 的权重
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("✅ 权重加载成功 (QAT FP32)")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        exit()

    # F. 执行转换 (FP32 -> INT8)
    print("正在转换为 INT8 模型...")
    model.eval()
    # 这一步是关键：它把模型变成了真正的 Quantized 模型，模拟手机端的行为
    model_int8 = torch.quantization.convert(model, inplace=False)
    print("✅ 转换完成！现在是真正的 INT8 模型。")

    return model_int8


# ================= 3. 实时推理 =================
def main():
    model = load_and_convert_model()

    # 预处理
    preprocess = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    print("\n>>> 开始推理 (按 'q' 退出) <<<")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. 图像处理: BGR -> RGB -> PIL
        # 这一步对应 Android 的 Bitmap 转换
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # 2. 预处理
        input_tensor = preprocess(pil_img).unsqueeze(0)

        # 3. 推理 (INT8 推理在 CPU 上)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            scores, indices = torch.topk(probs, 1)

        class_id = indices[0].item()
        score = scores[0].item()
        class_name = categories[class_id]

        # 4. 显示结果
        color = (0, 255, 0)
        # 如果置信度太低，给个提示
        if score < 0.5:
            display_text = f"Unsure... ({class_name} {score:.2f})"
            color = (0, 255, 255)  # 黄色
        else:
            display_text = f"{class_name}: {score:.2f}"
            if class_name == "battery": color = (0, 0, 255)  # 电池标红

        cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 0), 4)
        cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, color, 2)

        cv2.imshow('INT8 QAT Inference', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()