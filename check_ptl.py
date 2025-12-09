import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import cv2
import numpy as np
import os

# ================= 配置 =================
MODEL_PATH = "model12.pth"  # 你的原始权重
IMAGE_PATH = "garbage_classification/biological/biological57.jpg"  # 测试图片
NUM_CLASSES = 12
device = torch.device("cpu")  # 手机是 CPU，这里也用 CPU 测，保持一致

# 字母序字典 (必须与手机端一致)
categories = {
    0: 'battery', 1: 'biological', 2: 'brown-glass', 3: 'cardboard',
    4: 'clothes', 5: 'green-glass', 6: 'metal', 7: 'paper',
    8: 'plastic', 9: 'shoes', 10: 'trash', 11: 'white-glass'
}


# ================= 1. 模型定义 (原始 FP32) =================
class GarbageClassifier(nn.Module):
    def __init__(self, num_classes=12):
        super(GarbageClassifier, self).__init__()
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


def verify_trace_process():
    print(f"🔍 [1] 加载原始权重: {MODEL_PATH}")
    model = GarbageClassifier(num_classes=NUM_CLASSES).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("✅ 权重加载成功")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return

    print("\n🔍 [2] 执行 Trace (模拟手机端模型生成)...")
    # 创建一个和手机输入完全一样的虚拟 Tensor (1, 3, 224, 224)
    example_input = torch.rand(1, 3, 224, 224).to(device)

    try:
        # 【关键步骤】这就是把 PyTorch 代码变成手机指令的过程
        traced_model = torch.jit.trace(model, example_input)
        print("✅ Trace 成功！这就是手机端运行的逻辑本体。")
    except Exception as e:
        print(f"❌ Trace 失败: {e}")
        return

    print("\n🔍 [3] 运行真实图片测试...")
    if not os.path.exists(IMAGE_PATH):
        print(f"⚠️ 没找到 {IMAGE_PATH}，无法进行图片测试")
        return

    # 读取与预处理 (完全模拟 Android 端的 createScaledBitmap + TensorImageUtils)
    img_pil = Image.open(IMAGE_PATH).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img_pil).unsqueeze(0).to(device)

    # 推理
    # ... 推理部分 ...
    with torch.no_grad():
        output = traced_model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)

        # 【修改这里】先取第0个数据，去掉 Batch 维度，变成一维数组
        probs = probs[0]
        top3_prob, top3_catid = torch.topk(probs, 3)

    # 显示结果
    print("-" * 30)
    # 现在的 top3_catid 是一维的 [3]，可以直接取 index
    top_name = categories[top3_catid[0].item()]
    top_score = top3_prob[0].item() * 100
    print(f"🏆 预测结果: {top_name} ({top_score:.2f}%)")
    print("-" * 30)

    for i in range(3):
        idx = top3_catid[i].item()
        print(f"{i + 1}. {categories[idx]:<12}: {top3_prob[i].item() * 100:.2f}%")

    # 弹窗显示
    opencv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    opencv_img = cv2.resize(opencv_img, (500, 500))
    cv2.putText(opencv_img, f"{top_name}: {top_score:.1f}%", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Trace Test", opencv_img)
    print("\n按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    verify_trace_process()