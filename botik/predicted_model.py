import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
from torch.utils.data import DataLoader

# Загрузка данных
df = pd.read_excel('data/mainspacykeword.xlsx')

# Загрузка предварительно обученного токенизатора BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocessing_text_with_transformer(text):
    # Токенизация
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512)
    input_ids = tokens['input_ids'][0].tolist()

    # Обратное преобразование в текст
    cleaned_text = tokenizer.decode(input_ids)
    return cleaned_text

# Применение предобработки с трансформером
df['pr_txt_with_transformer'] = df['keywords_text'].apply(preprocessing_text_with_transformer)

from sklearn.preprocessing import LabelEncoder
# Инициализация и применение LabelEncoder к меткам классов
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['Уровень рейтинга'])

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(df['pr_txt_with_transformer'], df['label_encoded'], test_size=0.2, random_state=42)

# Преобразование текста в тензоры для BERT
train_tokens = tokenizer(list(X_train), padding=True, truncation=True, return_tensors='pt', max_length=512)
test_tokens = tokenizer(list(X_test), padding=True, truncation=True, return_tensors='pt', max_length=512)

y_train = np.array(y_train)
y_test = np.array(y_test)

train_dataset = TensorDataset(train_tokens.input_ids, train_tokens.attention_mask, torch.tensor(y_train))
test_dataset = TensorDataset(test_tokens.input_ids, test_tokens.attention_mask, torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False)

# Загрузка предварительно обученной модели BERT для классификации
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))


num_epochs = 1

# Проверяем доступность GPU и выбираем его, если возможно
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU доступен. Используется GPU.")
else:
    device = torch.device('cpu')
    print("GPU не доступен. Используется CPU.")

# Обучение модели
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)



for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False)

    for batch in progress_bar:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        input_ids = input_ids.to(torch.long)
        attention_mask = attention_mask.to(torch.long)
        labels = labels.to(torch.long)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'Loss': total_loss / len(progress_bar)})

    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

# Оценка модели
model.eval()
with torch.no_grad():
    y_true = []
    y_pred = []
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Обратное преобразование числовых предсказаний в текстовые метки классов
y_true_original = label_encoder.inverse_transform(y_true)
y_pred_original = label_encoder.inverse_transform(y_pred)

f1_micro = f1_score(y_true_original, y_pred_original, average='micro')
print(f'F1 Micro: {f1_micro}')