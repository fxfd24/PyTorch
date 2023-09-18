import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
data = None
# Функция для загрузки данных из CSV-файла
def load_data():
    global data
    file_path = input("CSV files, path/*.csv")
    if file_path:
        data = pd.read_csv(file_path)
        data['<DATE>'] = pd.to_datetime(data['<DATE>'], format='%Y%m%d')
        data.set_index('<DATE>', inplace=True)
        print(f'max look_back => {len(data)-3}')

# Функция для обработки данных и обучения модели
def train_model():
    global data
    print(data)
    if data is None:
        print('empty_data')
        return
    # Запрос у пользователя количества эпох и размера пакета
    epochs = int(input('epochs_entry'))
    batch_size = int(input('batch_size_entry'))
    look_back = int(input('look_back_entry'))
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
    print(f'Predicted OPEN: {last_data_unscaled[0][0]}\nPredicted HIGH: {last_data_unscaled[0][1]}\nPredicted LOW: {last_data_unscaled[0][2]}\nPredicted CLOSE: {last_data_unscaled[0][3]}')

load_data()
train_model()