# 导入必要的库
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import sys
import time
import torch  # PyTorch核心库，用于张量和神经网络
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器
from torch.utils.data import Dataset, DataLoader  # 数据加载工具
import torchvision.transforms as transforms  # 图像预处理变换
import torchvision.models as models  # 预训练模型库
from PIL import Image  # 图像处理库
from sklearn.model_selection import train_test_split  # 数据拆分工具
from sklearn.metrics import classification_report  # 分类评估报告
import re  # 正则表达式，用于字符串处理

# 检查GPU可用性，并设置设备（优先CUDA）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')  # 输出：Using device: cuda 或 cpu

print('setup successful!')  # 确认导入成功

# %% md  # Jupyter Markdown cell：定义常量部分
# Define Constants
# %%

# 图像尺寸设置：224x224是MobileNetV2的标准输入大小，提高尺寸未显著提升准确率
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3  # RGB通道

# 数据集路径：假设数据集在当前目录下的 "garbage_classification/" 文件夹中，每个子文件夹对应一类
base_path = "garbage_classification/"

# 12类垃圾类别字典：键为数字标签（0-11），值为类别名
categories = {0: 'paper', 1: 'cardboard', 2: 'plastic', 3: 'metal', 4: 'trash', 5: 'battery',
              6: 'shoes', 7: 'clothes', 8: 'green-glass', 9: 'brown-glass', 10: 'white-glass',
              11: 'biological'}

print('defining constants successful!')  # 确认常量定义

# %% md  # Markdown：创建DataFrame部分
# Create DataFrame
# %% md
# We want to create a data frame that has in one column the filenames of all our images and in the other column the corresponding category.
# We Open the directories in the dataset one by one, save the filenames in the filenames_list and add the corresponding category in the categories_list

# %%

# 函数：为文件名添加类别前缀（如 "paper104.jpg" -> "paper/paper104.jpg"），便于ImageDataGenerator使用
def add_class_name_prefix(df, col_name):
    df[col_name] = df[col_name].apply(lambda x: x[:re.search("\d", x).start()] + '/' + x)
    return df

# 列表：存储所有文件名和对应类别标签
filenames_list = []
categories_list = []

# 遍历每个类别文件夹，收集文件名和重复标签
for category in categories:
    filenames = os.listdir(base_path + categories[category])  # 列出文件夹中的图像文件
    filenames_list = filenames_list + filenames  # 追加文件名
    categories_list = categories_list + [category] * len(filenames)  # 追加相同标签（长度匹配文件数）

# 创建DataFrame：两列，filename（图像名）和category（数字标签）
df = pd.DataFrame({
    'filename': filenames_list,
    'category': categories_list
})

# 应用前缀函数
df = add_class_name_prefix(df, 'filename')

# 随机打乱DataFrame顺序，确保随机性
df = df.sample(frac=1).reset_index(drop=True)

print('number of elements = ', len(df))  # 输出数据集总样本数

# %%
# 显示DataFrame头部，便于检查数据结构
df.head()

# %%
# 随机显示一个样本图像：用于可视化检查数据
random_row = random.randint(0, len(df) - 1)
sample = df.iloc[random_row]
randomimage = Image.open(base_path + sample['filename'])  # 加载PIL图像
print(sample['filename'])  # 打印文件名
plt.imshow(randomimage)  # 显示图像
plt.show()  # 关闭图像窗口

# %% md  # Markdown：可视化类别分布
# Viusalize the Categories Distribution
# %%

df_visualization = df.copy()
# 将数字标签转换为类别名称
df_visualization['category'] = df_visualization['category'].apply(lambda x: categories[x])

# 绘制柱状图：显示每类垃圾图像数量分布
df_visualization['category'].value_counts().plot.bar(y='count', x='category')

plt.xlabel("Garbage Classes", labelpad=14)  # x轴标签
plt.ylabel("Images Count", labelpad=14)    # y轴标签
plt.title("Count of images per class", y=1.02)  # 标题
plt.show()  # 显示图表

# %% md  # Markdown：创建模型部分
# Create the model
# %% md
# The steps are:
# 1. Create an mobilenetv2 model without the last layer and load the ImageNet pretrained weights
# 2. Add a pre-processing layer
# 3. Add a pooling layer followed by a softmax layer at the end

# %%

# 函数：MobileNetV2预处理（归一化到ImageNet均值/标准差）
def mobilenetv2_preprocessing(img):
    # Normalize to [0,1] then to [-1,1] as per MobileNetV2（实际在transforms中处理，这里未用）
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (img - mean) / std
    return img

# 自定义模型类：继承nn.Module
class GarbageClassifier(nn.Module):
    def __init__(self, num_classes=12):
        super(GarbageClassifier, self).__init__()
        # 加载预训练MobileNetV2，去掉分类头（include_top=False）
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet.classifier = nn.Identity()  # 移除原分类器（Identity不改变输入）
        # 可选调整第一层卷积（如果输入通道不同，但RGB标准无需）
        # self.mobilenet.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # 冻结骨干网络参数（transfer learning，只训分类头）
        for param in self.mobilenet.parameters():
            param.requires_grad = False

        # 添加全局平均池化 + 全连接层（输出num_classes）
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.mobilenet.last_channel, num_classes)  # last_channel=1280（MobileNetV2特征维度）

    def forward(self, x):
        # 前向传播：特征提取 -> 池化 -> 展平 -> 分类
        x = self.mobilenet.features(x)  # 通过MobileNet特征提取器
        x = self.global_pool(x)         # 全局平均池化
        x = torch.flatten(x, 1)         # 展平为1D（batch, 1280）
        x = self.fc(x)                  # 全连接输出logits（未softmax）
        return x

# 实例化模型并移到设备
model = GarbageClassifier(num_classes=len(categories)).to(device)

# 定义损失函数（交叉熵，适合多分类）和优化器（Adam，只优化fc层）
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # lr=0.001学习率

print(model)  # 打印模型结构

# %% md  # Markdown：早停回调
# We will use the EarlyStopping call back to stop our training if the validation_accuray is not improving for a certain number of epochs.

# %%

# 自定义早停类：监控val_accuracy，如果patience个epoch未改善则停止，并恢复最佳权重
class EarlyStopping:
    def __init__(self, patience=2, min_delta=0.001, restore_best_weights=True):
        self.patience = patience  # 耐心值
        self.min_delta = min_delta  # 最小改善阈值
        self.restore_best_weights = restore_best_weights
        self.best_loss = None  # 最佳val_acc（注意：代码中用loss但实际传acc，需调整）
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_accuracy, model):
        # 注意：代码中best_loss实际用于acc（>比较），但命名误导；实际监控acc提升
        if self.best_loss is None:
            self.best_loss = val_accuracy
            self.save_checkpoint(model)
        elif val_accuracy > self.best_loss + self.min_delta:
            self.best_loss = val_accuracy
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True  # 触发停止
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()  # 保存当前权重

# 实例化早停（patience=2）
early_stop = EarlyStopping(patience=2, min_delta=0.001)

print('call back defined!')

# %% md  # Markdown：数据拆分
# Split the Data Set
# %% md
# We split the training set into three separate sets: 80% train, 10% val, 10% test

# %%

# 将数字标签替换为名称（用于后续DataLoader的y_col）
df["category"] = df["category"].replace(categories)  # 现在category列是字符串

# 拆分数据集：先80/20（train/val+test），再val+test 50/50
train_df, validate_df = train_test_split(df, test_size=0.2, random_state=42)
validate_df, test_df = train_test_split(validate_df, test_size=0.5, random_state=42)

# 重置索引
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]

print('train size = ', total_train, 'validate size = ', total_validate, 'test size = ', test_df.shape[0])

# %% md  # Markdown：自定义数据集
# Custom Dataset Class
# %%

# 自定义Dataset类：继承torch.utils.data.Dataset
class GarbageDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, is_train=False):
        # 将字符串类别转为数字编码（Categorical.codes返回int8数组）
        self.labels = pd.Categorical(dataframe['category']).codes
        self.file_names = dataframe['filename'].values
        self.root_dir = root_dir
        self.transform = transform  # 图像变换
        self.is_train = is_train
        # 类别到索引映射（用于后续转换）
        self.class_to_idx = {cat: idx for idx, cat in enumerate(sorted(set(dataframe['category'])))}

    def __len__(self):
        return len(self.file_names)  # 数据集长度

    def __getitem__(self, idx):
        # 加载图像路径
        img_path = os.path.join(self.root_dir, self.file_names[idx])
        image = Image.open(img_path).convert('RGB')  # 转为RGB
        label = self.labels[idx]  # 获取标签（数字）

        if self.transform:
            image = self.transform(image)  # 应用变换（Resize, Normalize等）

        # 修复dtype问题：确保label为torch.long（CrossEntropyLoss要求）
        label = torch.tensor(label, dtype=torch.long)

        return image, label  # 返回图像张量和标签

# 定义变换：训练集（无增强）和验证/测试集
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),  # 调整大小
    transforms.ToTensor(),  # 转为张量 [0,1]
    # 数据增强（注释掉；可启用以防过拟合）
    # transforms.RandomRotation(30),
    # transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet归一化
])

val_test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 创建DataLoader：批处理数据
batch_size = 64  # 批大小（根据GPU内存调整）

train_dataset = GarbageDataset(train_df, base_path, transform=train_transform, is_train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)  # 训练时shuffle

val_dataset = GarbageDataset(validate_df, base_path, transform=val_test_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

test_dataset = GarbageDataset(test_df, base_path, transform=val_test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)  # 测试时batch=1，便于逐样本评估

print('Data loaders created!')

# %% md  # Markdown：训练模型
# Train the model
# %% md
# We will first create the training data generator... (数据生成器已替换为DataLoader)

# %%

EPOCHS = 20  # 训练轮数
history = {'loss': [], 'categorical_accuracy': [], 'val_loss': [], 'val_categorical_accuracy': []}  # 记录历史

# 训练循环
for epoch in range(EPOCHS):
    # 训练阶段
    model.train()  # 设置模型为训练模式（启用Dropout等）
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:  # 遍历批次
        inputs, labels = inputs.to(device), labels.to(device)  # 移到GPU

        optimizer.zero_grad()  # 清零梯度
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()  # 累积损失
        _, predicted = torch.max(outputs.data, 1)  # 预测类别（argmax）
        total_train += labels.size(0)  # 总样本
        correct_train += (predicted == labels).sum().item()  # 正确预测数

    # 计算平均损失和准确率
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train
    history['loss'].append(train_loss)
    history['categorical_accuracy'].append(train_acc)

    # 验证阶段
    model.eval()  # 设置为评估模式
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():  # 无梯度计算，节省内存
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    # 计算验证平均
    val_loss = running_val_loss / len(val_loader)
    val_acc = 100 * correct_val / total_val
    history['val_loss'].append(val_loss)
    history['val_categorical_accuracy'].append(val_acc)

    # 打印进度
    print(f'Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    # 早停检查（注释掉；可启用）
    if early_stop(val_acc / 100, model):  # 传acc分数（0-1）
        print(f'Early stopping at epoch {epoch + 1}')
        break

print('Training complete!')  # 训练结束

# %%
# 保存模型权重（不包括完整模型结构）
torch.save(model.state_dict(), "model12.pth")
print('Model saved as model12.pth')

# %% md  # Markdown：可视化训练过程
# Visualize the training process

# %%

# 绘制损失和准确率曲线
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(history['loss'], color='b', label="Training loss")
ax1.plot(history['val_loss'], color='r', label="validation loss")
ax1.set_yticks(np.arange(0, 0.7, 0.1))
ax1.legend()
ax1.set_title('Model Loss')

ax2.plot(history['categorical_accuracy'], color='b', label="Training accuracy")
ax2.plot(history['val_categorical_accuracy'], color='r', label="Validation accuracy")
ax2.legend()
ax2.set_title('Model Accuracy')

plt.tight_layout()  # 调整布局
plt.show()  # 显示图表

# %% md  # Markdown：测试评估
# Evaluate the test
# %% md
# To evaluate the performance of our model we will create a test generator...

# %%

# 测试阶段：评估模型
model.eval()  # 评估模式
correct_test = 0
total_test = 0
all_preds = []  # 所有预测
all_labels = []  # 所有真实标签

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)  # 预测

        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())  # 收集预测（CPU转numpy）
        all_labels.extend(labels.cpu().numpy())

# 计算测试准确率
accuracy = 100 * correct_test / total_test
print('Accuracy on test set = ', round(accuracy, 2), '%')

# %%
# 生成类别映射：数字索引到名称（基于测试集唯一类别排序）
gen_label_map = {i: cat for i, cat in enumerate(sorted(set(test_df['category'])))}
print(gen_label_map)  # 输出映射字典

# 转换预测和标签为名称字符串
preds = [gen_label_map[p] for p in all_preds]
labels = [gen_label_map[l] for l in all_labels]

# 打印分类报告：precision, recall, f1-score 等
print(classification_report(labels, preds))