import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split

# 加载CSV数据
data = pd.read_csv('/Users/llc/learning/UVA/UVA-SP/SP/multi/res/concat.csv')  # 替换成你的数据文件路径

# 划分特征和标签
X = data[['Mean_X', 'Mean_Y', 'Mean_Z', 'Std_X', 'Std_Y', 'Std_Z', 'Sqrt_X', 'Sqrt_Y', 'Sqrt_Z',
          'Median_X', 'Median_Y', 'Median_Z',
          'Max_vx', 'Max_vy', 'Max_vz',
          'Max_ax', 'Max_ay', 'Max_az']].values
y = data['Label'].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将数据转换为PyTorch的Tensor
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)  # 使用LongTensor，因为标签是整数
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(18, 64)  # 输入特征6，隐藏层64
        self.fc2 = nn.Linear(64, 64)  # 隐藏层64，隐藏层64
        self.fc3 = nn.Linear(64, 4)  # 隐藏层64，输出2（因为有两个类别）

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = NeuralNetwork()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# 在测试集上评估模型
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    predicted = torch.argmax(test_outputs, dim=1)
    print(f"test_outputs={test_outputs}-y_test={predicted}")
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
