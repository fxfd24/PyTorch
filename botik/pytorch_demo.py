import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim

data = None

# ������� ��� �������� ������ �� CSV-�����
def load_data():
    global data
    file_path = input("���� � ����� ")
    if file_path:
        data = pd.read_csv(file_path)
        data['<DATE>'] = pd.to_datetime(data['<DATE>'], format='%Y%m%d')
        data.set_index('<DATE>', inplace=True)
        print(f'max look_back => {len(data)-3}')

# ����������� ������ ��������� ����
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # �������� ������������ ������ ��� ���������� ���������� ����
        return out

# ������� ��� ��������� ������ � �������� ������
def train_model():
    global data
    # print(data)
    if data is None:
        print('empty_data')
        return
    # ������ � ������������ ���������� ���� � ������� ������
    epochs = int(input('epochs_entry'))
    batch_size = int(input('batch_size_entry'))
    look_back = int(input('look_back_entry'))
    forecast_days = 1

    # �������� ����� � �������������� �������
    data.dropna(inplace=True)

    # ��������������� ������
    scaler = MinMaxScaler()
    data[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']] = scaler.fit_transform(data[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']])

    # �������� � �������� ������
    X = []
    y = []

    for i in range(len(data) - look_back - forecast_days + 1):
        features = data.iloc[i:i + look_back][['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']].values
        X.append(features)
        target_index = i + look_back + forecast_days - 1  # ������ �������� �������� ��� ������� ��������
        y.append(data.iloc[target_index]['<CLOSE>'])

    X = np.array(X)
    y = np.array(y)

    # �������������� ������ � ������� PyTorch
    X = torch.tensor(X, dtype=torch.float32).cuda()
    y = torch.tensor(y, dtype=torch.float32).cuda()

     # �������� ���������� ������
    input_size = 4  # ���������� ���������
    hidden_size = 64
    num_layers = 1
    output_size = 1
    model = SimpleRNN(input_size, hidden_size, num_layers, output_size)
    model.to(torch.device("cuda"))

    # ����������� ������� ������ � ������������
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.000001)

    # �������� ������
    for epoch in range(epochs):
        outputs = model(X)
        optimizer.zero_grad()
        loss = criterion(outputs, y.view(-1))  # �������� ����������� y � ����������� �������
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


    # # ���������� ������
    # torch.save(model.state_dict(), 'model.pth')

    # # �������� ����� � ��������� ���������� � ��������
    # model_description = f"Input Size: {input_size}\nHidden Size: {hidden_size}\nNum Layers: {num_layers}\nOutput Size: {output_size}"
    # with open('model_description.txt', 'w') as f:
    #     f.write(model_description)

    # # ������������� Git ����������� (���� ��� �� ��� ���������������)
    # import git
    # repo = git.Repo.init(path='.')

    # # ���������� � ������ ��������� � �����������
    # repo.index.add(['model.pth', 'model_description.txt'])
    # repo.index.commit('Add trained model and description')

    # # �������� ��������� �� GitHub
    # import os
    # import git
    # from git import Repo

    # github_username = 'fxfd24'
    # github_password = 'Fxfd24roman'

    # repo_dir = os.getcwd()
    # repo_url = f'https://github.com/{github_username}/PyTorch.git'

    # repo = Repo(repo_dir)

    # origin = repo.remote('origin')
    # origin_url = origin.url

    # if origin_url != repo_url:
    #     # ��������� URL ���������� ����������� �� URL ������ GitHub �����������
    #     origin.url = repo_url

    # # �������� ���������
    # origin.pull()

    # # �������� ���������
    # origin.push(username=github_username, password=github_password)

    # ������������ ����� �� ��������� ����
    last_data = data.iloc[-look_back:][['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']].values #������� ������ ��� ������������
    # print(last_data)

    # ��������������� ��������� ������
    last_data = scaler.transform(last_data)

    # �������������� � ������ PyTorch
    last_data = torch.tensor(last_data, dtype=torch.float32).view(1, look_back, -1).cuda()   # ������� ���������� ����� �������

    predicted_price = model(last_data)

    #������������ �� cpu
    predicted_price = predicted_price.cpu()

    # ���������� �������� �� �������
    predicted_price = predicted_price.view(-1).detach().numpy()  # ��������� ��� ���������� ������

    # �������� ��������������� ������������� ���
    min_max_range = scaler.data_max_ - scaler.data_min_
    predicted_price = predicted_price * min_max_range + scaler.data_min_

    # ����� ������������� ���
    print(f'Predicted OPEN: {predicted_price[0]:.2f}\nPredicted HIGH: {predicted_price[1]:.2f}\nPredicted LOW: {predicted_price[2]:.2f}\nPredicted CLOSE: {predicted_price[3]:.2f}')

load_data()
train_model()
