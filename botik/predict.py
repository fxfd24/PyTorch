from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


def predict(text, model, tfidf_vectorizer, decoder):
    encoded_text = tfidf_vectorizer.transform([text])
    input = xgb.DMatrix(encoded_text)
    pred = model.predict(input)
    pred = pred.astype('int')
    decoded_pred = decoder.inverse_transform(pred)[0]

    print(decoded_pred)
	
	
loaded_model = xgb.Booster()
loaded_model.load_model('xgb_model_last.model')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
encoder = joblib.load('label_encoder.pkl')
text = "example of text" ## Здесь будет текст из файла который ты загружаешь в функцию


# predict(df.loc[0, 'pr_txt'], loaded_model, tfidf_vectorizer, encoder) ## вместе df.loc надо будет передавать текст
predict(text, loaded_model, tfidf_vectorizer, encoder) ## вместе df.loc надо будет передавать текст