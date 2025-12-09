import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torch.quantization import QuantStub, DeQuantStub
# 引入高级量化配置工具
from torch.quantization import HistogramObserver, PerChannelMinMaxObserver, QConfig
from PIL import Image
import random

# ==========================================
# 1. 配置参数
# ==========================================
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 1
NUM_CLASSES = 12
MODEL_PATH = "model12.pth"
DATA_DIR = "garbage_classification/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 字母序字典 (必须与 Pandas 的排序逻辑一致)
categories = {
    0: 'battery', 1: 'biological', 2: 'brown-glass', 3: 'cardboard',
    4: 'clothes', 5: 'green-glass', 6: 'metal', 7: 'paper',
    8: 'plastic', 9: 'shoes', 10: 'trash', 11: 'white-glass'
}


# ==========================================
# 2. 数据集加载
# ==========================================
class GarbageDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform

        if not os.path.exists(root_dir):
            raise RuntimeError(f"找不到数据集文件夹: {root_dir}")

        name_to_idx = {v: k for k, v in categories.items()}

        temp_files = []
        temp_labels = []
        for cat_name in os.listdir(root_dir):
            cat_dir = os.path.join(root_dir, cat_name)
            if os.path.isdir(cat_dir) and cat_name in name_to_idx:
                label_id = name_to_idx[cat_name]
                for img_name in os.listdir(cat_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        temp_files.append(os.path.join(cat_dir, img_name))
                        temp_labels.append(label_id)

        # 简单划分 8:2
        total_len = len(temp_files)
        indices = list(range(total_len))
        random.seed(42)
        random.shuffle(indices)

        split_idx = int(total_len * 0.8)
        if mode == 'train':
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]

        self.file_list = [temp_files[i] for i in self.indices]
        self.labels = [temp_labels[i] for i in self.indices]
        print(f"[{mode}] 加载: {len(self.file_list)} 张")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.file_list[idx]).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224))

        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ==========================================
# 3. 模型结构与 QAT 修复
# ==========================================
# 修复 MobileNetV2 加法问题的模块
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


# ==========================================
# 4. 辅助函数
# ==========================================
def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.5):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1.0 - alpha) * hard_loss


def evaluate(model, loader, name="Test"):
    model.eval()
    correct = 0
    total = 0
    print(f"\n评估 {name} ...")
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"{name} Accuracy: {acc:.2f}%")
    return acc


# ==========================================
# 5. 主程序 (Main)
# ==========================================
def main():
    # 1. 加载数据
    train_loader = DataLoader(GarbageDataset(DATA_DIR, train_transform, 'train'), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4)
    test_loader = DataLoader(GarbageDataset(DATA_DIR, train_transform, 'test'), batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4)

    # 2. 加载 Teacher (FP32)
    print("\n[1/5] 加载 Teacher...")
    teacher_model = QuantizableGarbageClassifier(num_classes=NUM_CLASSES).to(device)
    teacher_model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    teacher_model.eval()
    evaluate(teacher_model, test_loader, "Teacher(FP32)")

    # 3. 初始化 Student
    print("\n[2/5] 初始化 Student...")
    student_model = QuantizableGarbageClassifier(num_classes=NUM_CLASSES).to(device)
    student_model.load_state_dict(teacher_model.state_dict())
    replace_with_qat_blocks(student_model.mobilenet)  # 替换加法模块

    # -------------------------------------------------------------
    # 【核心创新点】配置高级量化策略
    # -------------------------------------------------------------
    student_model.train()  # 必须先切到 train 模式

    # 策略 A: 激活值使用 Histogram (KL散度) 观测器
    # 作用：分析每一层输出的直方图，忽略离群点，保留主要信息
    act_observer = HistogramObserver.with_args(reduce_range=False)

    # 策略 B: 权重使用 Per-Channel (分通道) 观测器
    # 作用：给卷积核的每个 Channel 单独分配 Scale，精度更高
    weight_observer = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)

    # 将策略打包成 Config
    student_model.qconfig = QConfig(activation=act_observer, weight=weight_observer)

    # 插入伪量化节点 (这一步会把上面的策略应用到每一层)
    torch.quantization.prepare_qat(student_model, inplace=True)
    print(">>> 高级量化策略 (Histogram + Per-Channel) 已应用！")
    # -------------------------------------------------------------

    # 4. 微调训练
    print("\n[3/5] 开始 QAT 微调...")
    optimizer = optim.Adam(student_model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        student_model.train()
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_output = teacher_model(images)

            student_output = student_model(images)
            loss = distillation_loss(student_output, teacher_output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch {epoch + 1}, Step {i}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch + 1} Avg Loss: {total_loss / len(train_loader):.4f}")
        evaluate(student_model, test_loader, f"Student(QAT-Ep{epoch + 1})")
        torch.cuda.empty_cache()

    # 5. 导出
    print("\n[4/5] 导出模型...")
    # 保存 FP32 备份 (含伪量化节点)
    torch.save(student_model.state_dict(), "model_advanced_qat.pth")

    # 转成 INT8
    student_model.eval().cpu()
    quantized_model = torch.quantization.convert(student_model, inplace=False)

    # Trace 并优化
    example_input = torch.rand(1, 3, 224, 224)
    traced_model = torch.jit.trace(quantized_model, example_input)
    from torch.utils.mobile_optimizer import optimize_for_mobile
    traced_model_opt = optimize_for_mobile(traced_model)
    traced_model_opt._save_for_lite_interpreter("garbage_mobile_advanced.ptl")

    print("\n✅ 完成！已生成 garbage_mobile_advanced.ptl (使用了 KL散度 + Per-Channel 量化)")


if __name__ == "__main__":
    main()