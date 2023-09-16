import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import os  # Добавлен импорт модуля для работы с файловой системой

data = None

# Функция для загрузки данных из CSV-файла
def load_data():
    global data
    file_path = input('data path: ')
    if file_path:
        data = pd.read_csv(file_path)
        data['<DATE>'] = pd.to_datetime(data['<DATE>'], format='%Y%m%d')
        data.set_index('<DATE>', inplace=True)
        print(f'max look_back => {len(data)-3}')

# Определение класса нейронной сети
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
        out = self.fc(out[:, -1, :])  # Получаем предсказания только для последнего временного шага
        return out

# Функция для обработки данных и обучения модели
def train_model():
    global data
    # print(data)
    if data is None:
        print('empty_data')
        return
    # Запрос у пользователя количества эпох и размера пакета
    epochs = int(input('epochs_entry '))
    batch_size = int(input('batch_size_entry '))
    look_back = int(input('look_back_entry '))
    forecast_days = 1

    # Удаление строк с отсутствующими данными
    data.dropna(inplace=True)

    # Масштабирование данных
    scaler = MinMaxScaler()
    data[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']] = scaler.fit_transform(data[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']])

    # Создание и обучение модели
    X = []
    y = []

    for i in range(len(data) - look_back - forecast_days + 1):
        features = data.iloc[i:i + look_back][['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']].values
        X.append(features)
        target_index = i + look_back + forecast_days - 1  # Индекс целевого значения для текущей итерации
        y.append(data.iloc[target_index]['<CLOSE>'])

    X = np.array(X)
    y = np.array(y)

    # Преобразование данных в тензоры PyTorch
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Создание экземпляра модели
    input_size = 4  # Количество признаков
    hidden_size = 64
    num_layers = 1
    output_size = 1
    model = SimpleRNN(input_size, hidden_size, num_layers, output_size)

    # Определение функции потерь и оптимизатора
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Обучение модели
    for epoch in range(epochs):
        outputs = model(X)
        optimizer.zero_grad()
        loss = criterion(outputs, y.view(-1))  # Привести размерность y к одномерному тензору
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # Путь для сохранения модели
    model_save_dir = 'trained_model'
    os.makedirs(model_save_dir, exist_ok=True)  # Создаем папку, если она не существует
    model_save_path = os.path.join(model_save_dir, 'simple_rnn_model.pth')

    # Сохранение модели
    torch.save(model.state_dict(), model_save_path)

    print(f'Model saved to {model_save_path}')

load_data()
train_model()
