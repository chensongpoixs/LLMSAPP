'''@author: chensong
   @date: 2025-12-06 15:28:11
   @lasteditors: chensong
   @lastedittime: 2025-12-06 15:28:11
   @file_path: /  
   @description:      经典识别MNIST手写数字识别
'''
import torch; # 引入torch库
from torch import nn; # 从torch库中引入神经网络模块
from torch.utils.data import DataLoader; # 引入数据加载和数据集模块
from torchvision import datasets; # 引入图像转换模块
from torchvision.transforms import ToTensor; # 引入将图像转换为张量的模块


#引入训练数据集

import test


train_data = datasets.MNIST(
    root="./data", # 数据集存放路径
    train=True, # 指定为训练集
    download=True, # 如果数据集不存在则下载
    transform=ToTensor() # 将图像转换为张量    
) 

#引入测试数据集
test_data = datasets.MNIST(
    root="./data", # 数据集存放路径 
    train=False, # 指定为测试集
    download=True, # 如果数据集不存在则下载
    transform=ToTensor() # 将图像转换为张量
)

# 加载数据
# 设置梯度下降的参数
batch_size = 64; # 每个批次的样本数量

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True); # 创建训练数据加载器
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False); # 创建测试数据加载器

for X, y in train_dataloader: # 遍历测试数据加载器
    #   torch.Size([64, 1, 28, 28]) ==>  64个样本,1个通道,28x28像素
    print("Shape of X [N, C, H, W]: ", X.shape); # 打印输入数据的形状
    # torch.Size([64]) torch.int64  ==> 64个标签,数据类型为int64
    print("Shape of y: ", y.shape, y.dtype); # 打印标签的形状和数据类型
    break; # 只查看第一个批次的数据

for X, y in test_dataloader: # 遍历测试数据加载器
    print("Shape of X [N, C, H, W]: ", X.shape); # 打印输入数据的形状
    print("Shape of y: ", y.shape, y.dtype); # 打印标签的形状和数据类型
    break; # 只查看第一个批次的数据



from zmq import device


device = "cuda" if torch.cuda.is_available() else "cpu"; # 检查是否有可用的GPU,否则使用CPU
print("Using {} device".format(device)); # 打印使用的设备类型

 

# 构建神经网络类  
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__(); # 调用父类的构造函数
        self.flatten = nn.Flatten(); # 定义一个展平层
        self.linear_relu_stack = nn.Sequential( # 定义一个顺序容器
            nn.Linear(28*28, 128), # 全连接层,输入大小为28*28,输出大小为128
            nn.ReLU(), # ReLU激活函数
            nn.Linear(128, 128), # 全连接层,输入大小为128,输出大小为128
            nn.ReLU(), # ReLU激活函数
            nn.Linear(128, 10) # 全连接层,输入大小为128,输出大小为10
        );

    def forward(self, x):
        x = self.flatten(x); # 展平输入数据
        logits = self.linear_relu_stack(x); # 通过顺序容器进行前向传播
        return logits; # 返回输出结果
# 创建神经网络实例并移动到指定设备
model = NeuralNetwork().to(device);
print(model); # 打印神经网络结构



#定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss(); # 交叉熵损失函数,适用于多分类问题
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3); # 随机梯度下降优化器,学习率为0.001

# 训练神经网络tran
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset); # 获取数据集的大小
    model.train(); # 设置模型为训练模式
    for batch, (X, y) in enumerate(dataloader): # 遍历数据加载器
        X, y = X.to(device), y.to(device); # 将数据和标签移动到指定设备

        # 计算预测误差
        pred = model(X); # 通过模型进行前向传播
        loss = loss_fn(pred, y); # 计算损失

        # 反向传播
        optimizer.zero_grad(); # 清除梯度
        loss.backward(); # 反向传播计算梯度
        optimizer.step(); # 更新模型参数

        if batch % 100 == 0: # 每100个批次打印一次日志
            loss, current = loss.item(), batch * len(X); # 获取当前损失和处理的数据量
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]"); # 打印损失和进度信息





# 测试神经网络
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset); # 获取数据集的大小
    num_batches = len(dataloader); # 获取批次数量
    model.eval(); # 设置模型为评估模式
    test_loss, correct = 0, 0; # 初始化测试损失和正确预测数量
    with torch.no_grad(): # 禁用梯度计算
        for X, y in dataloader: # 遍历数据加载器
            X, y = X.to(device), y.to(device); # 将数据和标签移动到指定GPU设备
            pred = model(X); # 通过模型进行前向传播
            test_loss += loss_fn(pred, y).item(); # 累加损失
            correct += (pred.argmax(1) == y).type(torch.float).sum().item(); # 计算正确预测数量
    test_loss /= num_batches; # 计算平均损失
    correct /= size; # 计算准确率
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"); # 打印测试结果 
# 运行训练和测试循环
epochs = 50; # 设置训练轮数
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------");
    train(train_dataloader, model, loss_fn, optimizer); # 训练模型
    test(test_dataloader, model, loss_fn); # 测试模型
print("Done!");




# 模型保存
torch.save(model.state_dict(), "model_weights.pth"); # 保存模型参数到文件
print("Saved PyTorch Model State to model_weights.pth");



# 载入模型参数
model = NeuralNetwork().to(device);
model.load_state_dict(torch.load("model_weights.pth", weights_only=True) ); # 从文件加载模型参数
print("Loaded PyTorch Model State from model_weights.pth"); 




# 预测结果
model.eval(); # 设置模型为评估模式
x, y = test_data[0][0], test_data[0][1]; # 获取测试数据的第一个样本和标签
with torch.no_grad(): # 禁用梯度计算
    pred = model(x); # 通过模型进行前向传播
    predicted, actual = pred[0].argmax(0), y; # 获取预测结果和实际标签
    print(f"Predicted: {predicted}, Actual: {actual}"); # 打印预测结果和实际标签




# 打印 mathplotlib 版本
import matplotlib.pyplot as plt;
# print(f"matplotlib version: {plt.__version__}");

plt.imshow(x.cpu().squeeze(), cmap="gray"); # 显示图像
plt.title(f"Predicted: {predicted}, Actual: {actual}"); # 设置
plt.show(); # 显示图形