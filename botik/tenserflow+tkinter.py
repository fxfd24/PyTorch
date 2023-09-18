import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tqdm import tqdm

#https://www.finam.ru/profile/tovary/brent/export/

data = None

# Функция для загрузки данных из CSV-файла
def load_data():
    global data
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        data = pd.read_csv(file_path)
        # Преобразование даты в индекс
        data['<DATE>'] = pd.to_datetime(data['<DATE>'], format='%Y%m%d')
        data.set_index('<DATE>', inplace=True)
        print(f'max look_back => {len(data)-3}')
        # return data
    # return None

# Функция для обработки данных и обучения модели
def train_model():
    global data
    # data = load_data()
    print(data)
    if data is None:
        print('empty_data')
        return
    # Запрос у пользователя количества эпох и размера пакета
    epochs = int(epochs_entry.get())
    batch_size = int(batch_size_entry.get())
    look_back = int(look_back_entry.get())

    # Создание временных рядов и целевой переменной (курс на следующий день)
    #look_back = 360  # Количество дней для использования в качестве признаков
    forecast_days = 1  # Количество дней для предсказания вперед (1 день)

    # Удаление строк с отсутствующими данными
    data.dropna(inplace=True)

    # Масштабирование данных
    scaler = MinMaxScaler()
    data[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']] = scaler.fit_transform(data[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']])

    # print(data)
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

    # Создание и обучение модели
    model = tf.keras.Sequential([

        tf.keras.layers.Dense(64, activation='relu', input_shape=(look_back, 4)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # 1 выходной нейрон для предсказания цены на следующий день
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')


    # Использование прогресс-бара во время обучения
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=2)

    # Предполагая, что X был сформирован правильно, убедитесь, что его форма правильная
    print(X.shape)  # Должно быть (количество_образцов, 360, 4)

    # Проверьте, что X имеет правильную форму
    assert X.shape == (len(data) - look_back - forecast_days + 1, look_back, 4)

    ## Предсказание курса на следующий день
    last_data = data.iloc[-look_back:][['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']].values.reshape(1, look_back, 4)
    predicted_price = model.predict(last_data)

    # Обратное масштабирование предсказанных цен обратно к оригинальным признакам
    real_last_data = scaler.inverse_transform(last_data.reshape(look_back, 4))[:-1]  # Исключаем последний день из последовательности
    predicted_price_reshaped = predicted_price.reshape(-1, 4)  # Изменяем форму предсказаний
    predicted_price_unscaled = scaler.inverse_transform(predicted_price_reshaped)

    # Объединение массивов
    real_predicted_price = np.concatenate((real_last_data, predicted_price_unscaled), axis=0)
    # Масштабирование предсказанных цен обратно к оригинальным признакам
    last_data_unscaled = scaler.inverse_transform(last_data.reshape(-1, 4))

    # Получение даты следующего дня
    next_day = data.index[-1] + pd.DateOffset(days=1)

    # Вывод реальной предсказанной цены на следующий день с указанием даты и других параметров
    result_label.config(text=f'Predicted OPEN: {last_data_unscaled[0][0]}\n'
                             f'Predicted HIGH: {last_data_unscaled[0][1]}\n'
                             f'Predicted LOW: {last_data_unscaled[0][2]}\n'
                             f'Predicted CLOSE: {last_data_unscaled[0][3]}')


# Создание графического интерфейса
root = tk.Tk()
root.title("Stock Price Prediction")

frame = ttk.Frame(root)
frame.grid(column=0, row=0, padx=10, pady=10)

# Виджеты для ввода параметров
epochs_label = ttk.Label(frame, text="Количество эпох для обучения:")
epochs_label.grid(column=0, row=0, padx=5, pady=5)

epochs_entry = ttk.Entry(frame)
epochs_entry.grid(column=1, row=0, padx=5, pady=5)
epochs_entry.insert(0, "50")

batch_size_label = ttk.Label(frame, text="Размер пакета (batch size):")
batch_size_label.grid(column=0, row=1, padx=5, pady=5)

batch_size_entry = ttk.Entry(frame)
batch_size_entry.grid(column=1, row=1, padx=5, pady=5)
batch_size_entry.insert(0, "32")

# data_ = pd.read_csv(file_path)
look_back_label = ttk.Label(frame, text=f"Укажите период для train:")
look_back_label.grid(column=0, row=2, padx=5, pady=5)

look_back_entry = ttk.Entry(frame)
look_back_entry.grid(column=1, row=2, padx=5, pady=5)
look_back_entry.insert(0, "360")

# Кнопка для загрузки данных и обучения модели
load_button = ttk.Button(frame, text="Загрузить данные", command=load_data)
load_button.grid(column=0, row=3, columnspan=2, padx=5, pady=10)

train_button = ttk.Button(frame, text="Обучить модель", command=train_model)
train_button.grid(column=0, row=4, columnspan=2, padx=5, pady=10)

# Виджет для вывода результата
result_label = ttk.Label(frame, text="", font=("Helvetica", 12))
result_label.grid(column=0, row=5, columnspan=2, padx=5, pady=10)

root.geometry("500x400")
root.mainloop()
