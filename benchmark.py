import torch
import torch.nn as nn
import torchvision.models as models
import time
import os

# ---------------------------------------------------------
# 1. 模型定义 (直接复用你的代码)
# ---------------------------------------------------------
class GarbageClassifier(nn.Module):
    def __init__(self, num_classes=12):
        super(GarbageClassifier, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=False) # 测试时不需要下载预训练权重结构
        self.mobilenet.classifier = nn.Identity()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.mobilenet.last_channel, num_classes)

    def forward(self, x):
        x = self.mobilenet.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ---------------------------------------------------------
# 2. 准备工作
# ---------------------------------------------------------
# 设定设备：边缘设备通常使用 CPU (如树莓派) 或 特定 GPU (如 Jetson)
# 这里我们先在 CPU 上测，因为很多边缘推理是在 CPU 上进行的
device = torch.device("cpu")
print(f"正在测试设备: {device}")

# 加载模型
model = GarbageClassifier(num_classes=12).to(device)

# 如果你有训练好的权重，加载进去（虽然权重值不影响速度，但为了严谨可以加载）
# try:
#     model.load_state_dict(torch.load("model12.pth", map_location=device))
# except:
#     print("未加载权重，使用随机初始化进行基准测试...")

model.eval() # 切换到评估模式

# ---------------------------------------------------------
# 3. 测量参数量 (Model Size)
# ---------------------------------------------------------
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n[1] 模型大小:")
print(f"总参数量: {total_params / 1e6:.2f} M (百万)")
print(f"模型权重文件预估大小: {total_params * 4 / (1024*1024):.2f} MB (FP32精度)")

# ---------------------------------------------------------
# 4. 测量计算量 (FLOPs) - 需要安装 thop 库
# ---------------------------------------------------------
try:
    from thop import profile
    # 创建一个虚拟输入 (Batch=1, RGB=3, 224x224)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(model, inputs=(dummy_input, ), verbose=False)
    print(f"\n[2] 计算复杂度:")
    print(f"FLOPs: {flops / 1e6:.2f} M (百万次浮点运算)")
except ImportError:
    print("\n[2] 计算复杂度: 请安装 'thop' 库以查看 FLOPs (pip install thop)")

# ---------------------------------------------------------
# 5. 测量推理速度 (Latency & FPS)
# ---------------------------------------------------------
# 创建虚拟输入
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# 预热 (Warm-up): 让 CPU/GPU 进入工作状态，避免第一次运行慢的影响
print("\n正在预热模型...")
with torch.no_grad():
    for _ in range(10):
        _ = model(dummy_input)

# 正式测试
iterations = 100 # 测试 100 次取平均
print(f"开始测试 (运行 {iterations} 次)...")

start_time = time.time()
with torch.no_grad():
    for _ in range(iterations):
        _ = model(dummy_input)
end_time = time.time()

total_time = end_time - start_time
avg_latency = (total_time / iterations) * 1000 # 毫秒
fps = 1 / (total_time / iterations)

print(f"\n[3] 推理性能 (纯模型，不含预处理/后处理):")
print(f"平均延迟 (Latency): {avg_latency:.2f} ms")
print(f"帧率 (FPS): {fps:.2f} frames/sec")