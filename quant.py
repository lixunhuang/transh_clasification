import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torch.quantization import QuantStub, DeQuantStub
from PIL import Image
import copy

# ==========================================
# 1. 配置参数
# ==========================================
BATCH_SIZE = 8  # 保持 8 以防显存爆炸
LEARNING_RATE = 1e-4  # 微调学习率
EPOCHS = 5  # 微调轮数
NUM_CLASSES = 12
MODEL_PATH = "model12.pth"  # 你的原始权重
DATA_DIR = "garbage_classification/"  # 数据集路径

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 字母序字典
categories = {
    0: 'battery', 1: 'biological', 2: 'brown-glass', 3: 'cardboard',
    4: 'clothes', 5: 'green-glass', 6: 'metal', 7: 'paper',
    8: 'plastic', 9: 'shoes', 10: 'trash', 11: 'white-glass'
}


# ==========================================
# 2. 数据集 (带划分)
# ==========================================
class GarbageDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.all_files = []
        self.all_labels = []

        if not os.path.exists(root_dir):
            raise RuntimeError(f"找不到数据集文件夹: {root_dir}")

        name_to_idx = {v: k for k, v in categories.items()}

        # 遍历所有图片
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

        # 简单的手动划分：80% 训练，20% 测试
        # 为了保证每次划分一致，这里不随机打乱，或者设定固定种子
        total_len = len(temp_files)
        split_idx = int(total_len * 0.8)

        # 简单切分 (实际项目最好先 shuffle 再切，或者用 sklearn)
        # 这里为了演示简单，直接按顺序切可能会导致某些类别全是训练集，所以我们手动 shuffle 一下索引
        indices = list(range(total_len))
        import random
        random.seed(42)  # 固定种子
        random.shuffle(indices)

        if mode == 'train':
            self.indices = indices[:split_idx]
        else:  # test
            self.indices = indices[split_idx:]

        self.file_list = [temp_files[i] for i in self.indices]
        self.labels = [temp_labels[i] for i in self.indices]

        print(f"[{mode.upper()}] 数据集加载完成: {len(self.file_list)} 张")

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
# 3. 关键修复：定义支持量化的 Residual Block
# ==========================================
# 这个类用来替换 MobileNetV2 里原始的 InvertedResidual
class QATInvertedResidual(nn.Module):
    def __init__(self, original_block):
        super().__init__()
        # 直接偷取原始 block 的卷积层，共享权重！
        self.conv = original_block.conv
        self.use_res_connect = original_block.use_res_connect
        # 【关键】使用 FloatFunctional 处理加法
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            # 原始代码是: return x + self.conv(x)
            # QAT代码是:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)


# 递归替换函数
def replace_with_qat_blocks(model):
    for name, child in model.named_children():
        if type(child).__name__ == 'InvertedResidual':
            # 发现目标，进行替换
            setattr(model, name, QATInvertedResidual(child))
        else:
            # 递归查找
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
# 4. 蒸馏与评估函数
# ==========================================
def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.5):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1.0 - alpha) * hard_loss


def evaluate(model, loader, device_name="Test"):
    model.eval()
    correct = 0
    total = 0
    print(f"\n正在评估 {device_name} 集...")
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"{device_name} Accuracy: {acc:.2f}%")
    return acc


# ==========================================
# 5. 主程序
# ==========================================
def main():
    # 1. 准备数据
    train_dataset = GarbageDataset(DATA_DIR, transform=train_transform, mode='train')
    test_dataset = GarbageDataset(DATA_DIR, transform=train_transform, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. 准备 Teacher (FP32)
    print("\n[Step 1] 加载 Teacher 模型...")
    teacher_model = QuantizableGarbageClassifier(num_classes=NUM_CLASSES).to(device)
    teacher_model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    teacher_model.eval()

    # 评估一下 Teacher 的基准水平
    print(">>> Teacher 模型基准测试:")
    evaluate(teacher_model, test_loader, "Teacher(FP32)")

    # 3. 准备 Student (QAT)
    print("\n[Step 2] 初始化 Student 模型...")
    student_model = QuantizableGarbageClassifier(num_classes=NUM_CLASSES).to(device)
    # 继承权重
    student_model.load_state_dict(teacher_model.state_dict())

    # 【关键步骤】替换网络中的“坏块”为支持量化的“好块”
    # 必须在 load_state_dict 之后做，这样权重自然就带过去了
    replace_with_qat_blocks(student_model.mobilenet)
    print("已替换 MobileNetV2 的 InvertedResidual 模块为 QAT 兼容版。")

    # 配置 QAT
    student_model.train()
    student_model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    torch.quantization.prepare_qat(student_model, inplace=True)
    print("Student 模型已进入 QAT 模式。")

    # 4. 训练 (QAT + 蒸馏)
    optimizer = optim.Adam(student_model.parameters(), lr=LEARNING_RATE)

    print("\n[Step 3] 开始 QAT 微调...")
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

            if i % 50 == 0:
                print(f"Epoch {epoch + 1}, Step {i}, Loss: {loss.item():.4f}")

        # 每个 Epoch 结束后评估一次
        print(f"Epoch {epoch + 1} 结束。平均 Loss: {total_loss / len(train_loader):.4f}")
        evaluate(student_model, test_loader, "Student(QAT-模拟)")

        # 简单清理显存
        torch.cuda.empty_cache()

    # 5. 保存 QAT 权重 (.pth) - 这是你要的改进版权重
    print("\n[Step 4] 保存 QAT 训练后的权重 (.pth)...")
    # 注意：保存前最好把 quant/dequant 变为 eval 模式，或者直接存
    # 这里我们存的是带有伪量化节点的 FP32 模型，非常适合后续分析
    torch.save(student_model.state_dict(), "model12_qat_improved.pth")
    print("已保存: model12_qat_improved.pth (可用于对比 Baseline)")

    # 6. 导出 INT8 模型 (.ptl)
    print("\n[Step 5] 导出最终 INT8 手机模型...")
    student_model.eval().cpu()

    # 真正的转换
    quantized_model = torch.quantization.convert(student_model, inplace=False)

    # Trace
    example_input = torch.rand(1, 3, 224, 224)
    # 因为我们替换了 block，现在的 forward 包含 FloatFunctional，是可以 trace 的
    traced_model = torch.jit.trace(quantized_model, example_input)

    # 手机优化
    from torch.utils.mobile_optimizer import optimize_for_mobile
    traced_model_opt = optimize_for_mobile(traced_model)

    traced_model_opt._save_for_lite_interpreter("garbage_mobile_int8_qat.ptl")

    print("\n✅ 全部完成！")
    print(f"1. 改进版权重: model12_qat_improved.pth (FP32, QAT trained)")
    print(f"2. 手机端模型: garbage_mobile_int8_qat.ptl (INT8)")


if __name__ == "__main__":
    main()