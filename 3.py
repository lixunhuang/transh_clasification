import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.mobile_optimizer import optimize_for_mobile
import os

# ================= 配置 =================
MODEL_PATH = "model12.pth"  # 你的原始权重
SAVE_NAME = "garbage_mobile_final.ptl"  # 最终成品
NUM_CLASSES = 12
device = torch.device("cpu")


# ================= 1. 最原始的模型定义 (匹配 model12.pth) =================
class GarbageClassifier(nn.Module):
    def __init__(self, num_classes=12):
        super(GarbageClassifier, self).__init__()
        # 你的训练代码里用了 pretrained=True，这里用 False 也可以，因为会加载权重
        # 但结构必须一致
        self.mobilenet = models.mobilenet_v2(pretrained=False)
        self.mobilenet.classifier = nn.Identity()

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.mobilenet.last_channel, num_classes)

    def forward(self, x):
        # 没有任何量化节点，纯纯的 FP32 计算
        x = self.mobilenet.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ================= 2. 转换 =================
def convert():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到 {MODEL_PATH}")
        return

    print(f"正在加载 {MODEL_PATH} ...")
    model = GarbageClassifier(num_classes=NUM_CLASSES).to(device)

    try:
        # 加载权重
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ 权重加载成功！")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    # Trace (追踪)
    print("正在转换为移动端模型 (Tracing)...")
    example_input = torch.rand(1, 3, 224, 224)
    try:
        traced_script_module = torch.jit.trace(model, example_input)

        # 移动端优化 (这一步虽然不是量化，但能显著提升 APP 里的运行速度)
        traced_script_module_optimized = optimize_for_mobile(traced_script_module)

        # 保存
        traced_script_module_optimized._save_for_lite_interpreter(SAVE_NAME)
        print(f"\n🎉 大功告成！文件已生成: {SAVE_NAME}")
        print(f"文件大小: {os.path.getsize(SAVE_NAME) / 1024 / 1024:.2f} MB")

    except Exception as e:
        print(f"❌ 转换失败: {e}")


if __name__ == "__main__":
    convert()