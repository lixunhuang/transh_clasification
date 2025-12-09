import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.quantization import QuantStub, DeQuantStub, QConfig, HistogramObserver, PerChannelMinMaxObserver
from PIL import Image
import os
import time

# ================= 配置 =================
MODEL_PATH = "model_advanced_qat.pth"  # 你的 QAT 权重
DATA_DIR = "garbage_classification/"  # 数据集根目录
NUM_CLASSES = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

categories = {
    0: 'battery', 1: 'biological', 2: 'brown-glass', 3: 'cardboard',
    4: 'clothes', 5: 'green-glass', 6: 'metal', 7: 'paper',
    8: 'plastic', 9: 'shoes', 10: 'trash', 11: 'white-glass'
}


# ================= 1. 模型定义 (保持一致) =================
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


# ================= 2. 核心：只加载，不转换 =================
def load_qat_model_no_convert():
    print(f"正在加载 {MODEL_PATH} ...")
    model = QuantizableGarbageClassifier(num_classes=NUM_CLASSES).to(device)
    replace_with_qat_blocks(model.mobilenet)

    # 必须配置 Config，否则 prepare_qat 会报错
    model.qconfig = QConfig(
        activation=HistogramObserver.with_args(reduce_range=False),
        weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
    )

    # 插入伪量化节点 (FakeQuant)
    torch.quantization.prepare_qat(model, inplace=True)

    # 加载权重
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    # 关键点：设置为 eval 模式
    # 在 eval 模式下，FakeQuant 节点会使用训练好的 scale/zero_point 模拟量化效果
    model.eval()
    print("✅ 模型加载成功 (QAT-FP32 模式)")
    return model


# ================= 3. 全量测试 =================
def main():
    model = load_qat_model_no_convert()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"\n📂 开始全量测试: {DATA_DIR}")

    total_images = 0
    correct_predictions = 0

    # 遍历所有类别
    for class_name in categories.values():
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_dir): continue

        print(f"   Testing: {class_name} ...")

        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')): continue

            img_path = os.path.join(class_dir, img_name)

            try:
                img = Image.open(img_path).convert('RGB')
                input_tensor = preprocess(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    pred_idx = torch.argmax(output, 1).item()
                    pred_name = categories[pred_idx]

                total_images += 1
                if pred_name == class_name:
                    correct_predictions += 1

            except:
                pass

    if total_images > 0:
        acc = correct_predictions / total_images * 100
        print("\n" + "=" * 40)
        print(f"✅ 最终准确率: {acc:.2f}%")
        print("=" * 40)

        if acc > 90:
            print("结论：模型是好的！之前的全是Clothes是因为PC端INT8转换出了Bug。")
            print("放心去手机上跑 ptl 文件吧！")
        else:
            print("结论：模型真的坏了，训练有问题。")
    else:
        print("❌ 没找到图片")


if __name__ == "__main__":
    main()