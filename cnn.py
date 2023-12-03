
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
X_train = X_train.unsqueeze(1)  # 将数据从[批大小, 特征数]转换为[批大小, 1, 特征数]
X_test = X_test.unsqueeze(1)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(4 * 32, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = SimpleCNN(num_classes=4)  # 假设您有4个类别


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
