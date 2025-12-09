import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.mobile_optimizer import optimize_for_mobile

# 1. 定义和之前一模一样的模型结构
class GarbageClassifier(nn.Module):
    def __init__(self, num_classes=12):
        super(GarbageClassifier, self).__init__()
        # 加载 MobileNetV2 (不需要下载预训练权重，因为我们会加载你存好的)
        self.mobilenet = models.mobilenet_v2(pretrained=False)
        self.mobilenet.classifier = nn.Identity()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.mobilenet.last_channel, num_classes)

    def forward(self, x):
        x = self.mobilenet.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 2. 加载模型和权重
device = torch.device("cpu")
model = GarbageClassifier(num_classes=12).to(device)

try:
    # 加载你的权重文件
    model.load_state_dict(torch.load("model12.pth", map_location=device))
    model.eval() # 非常重要！
    print("权重加载成功！")
except Exception as e:
    print(f"错误：找不到 model12.pth，请检查文件路径。报错信息: {e}")
    exit()

# 3. 转换模型 (Tracing)
# 创建一个假的输入数据，尺寸必须是 (1, 3, 224, 224)
example_input = torch.rand(1, 3, 224, 224)

try:
    # 追踪模型
    traced_script_module = torch.jit.trace(model, example_input)
    # 针对手机优化
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    # 保存文件
    traced_script_module_optimized._save_for_lite_interpreter("garbage_mobile.ptl")
    print("大功告成！已生成 'garbage_mobile.ptl'，请把它发送到你的电脑桌面备用。")
except Exception as e:
    print(f"转换失败: {e}")