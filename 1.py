import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.quantization import QuantStub, DeQuantStub, QConfig, HistogramObserver, PerChannelMinMaxObserver
from PIL import Image
import os
import time

# ================= 配置 =================
MODEL_PATH = "model_advanced_qat.pth"  # 你的 FP32(QAT) 权重文件
DATA_DIR = "garbage_classification/"  # 数据集根目录
NUM_CLASSES = 12

# 字母序字典 (反向映射用于统计)
categories = {
    0: 'battery', 1: 'biological', 2: 'brown-glass', 3: 'cardboard',
    4: 'clothes', 5: 'green-glass', 6: 'metal', 7: 'paper',
    8: 'plastic', 9: 'shoes', 10: 'trash', 11: 'white-glass'
}
# 名字转ID
name_to_idx = {v: k for k, v in categories.items()}


# ================= 1. 模型结构定义 (保持不变) =================
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


# ================= 2. 模型加载与转换 =================
def load_and_convert_to_int8():
    print(f"🔄 正在准备量化环境...")

    # 【关键】强制设置量化后端为 qnnpack (模拟手机环境)
    # 这一步是为了验证你的 QConfig (针对 qnnpack 优化) 在 PC 上是否能跑通
    torch.backends.quantized.engine = 'qnnpack'

    device = torch.device("cpu")
    model = QuantizableGarbageClassifier(num_classes=NUM_CLASSES).to(device)
    replace_with_qat_blocks(model.mobilenet)

    # 配置 QConfig (必须与训练代码一致)
    model.qconfig = QConfig(
        activation=HistogramObserver.with_args(reduce_range=False),
        weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
    )

    torch.quantization.prepare_qat(model, inplace=True)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到权重文件 {MODEL_PATH}")
        exit()

    # 加载 FP32 权重
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    # 转换为 INT8
    model.eval()
    print("⚡ 正在执行 FP32 -> INT8 转换...")
    model_int8 = torch.quantization.convert(model, inplace=False)
    print("✅ 模型转换完成！")

    return model_int8


# ================= 3. 全量测试主程序 =================
def main():
    model = load_and_convert_to_int8()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"\n📂 开始扫描数据集: {DATA_DIR}")

    total_images = 0
    correct_predictions = 0

    # 记录每个类别的准确率
    class_correct = {cat: 0 for cat in categories.values()}
    class_total = {cat: 0 for cat in categories.values()}

    start_time = time.time()

    # 遍历所有类别文件夹
    for class_name in categories.values():
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        print(f"   正在测试类别: {class_name} ...")

        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(class_dir, img_name)

            try:
                # 1. 读取
                img = Image.open(img_path).convert('RGB')
                # 2. 预处理
                input_tensor = preprocess(img).unsqueeze(0)

                # 3. 推理
                with torch.no_grad():
                    output = model(input_tensor)
                    # 获取预测结果
                    pred_idx = torch.argmax(output, 1).item()
                    pred_name = categories[pred_idx]

                # 4. 统计
                total_images += 1
                class_total[class_name] += 1

                if pred_name == class_name:
                    correct_predictions += 1
                    class_correct[class_name] += 1

            except Exception as e:
                print(f"   ⚠️ 读取失败: {img_path}")

    end_time = time.time()
    total_time = end_time - start_time

    # ================= 4. 输出报告 =================
    print("\n" + "=" * 40)
    print("📊 INT8 模型全量测试报告")
    print("=" * 40)
    print(f"总图片数: {total_images}")
    print(f"总耗时:   {total_time:.2f} 秒")
    print(f"平均速度: {total_time / total_images * 1000:.1f} ms/张 (CPU单线程)")
    print("-" * 40)

    if total_images > 0:
        overall_acc = correct_predictions / total_images * 100
        print(f"🏆 总体准确率: {overall_acc:.2f}%")
        print("-" * 40)
        print("各类别准确率详情:")
        for cat_name in categories.values():
            count = class_total[cat_name]
            if count > 0:
                acc = class_correct[cat_name] / count * 100
                print(f"   {cat_name:<12}: {acc:.2f}% ({class_correct[cat_name]}/{count})")
    else:
        print("❌ 未找到任何图片，请检查 DATA_DIR 路径")


if __name__ == "__main__":
    main()