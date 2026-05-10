import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_data(df):
    df = df.dropna()

    print("Columns:", df.columns)

    # Correct columns
    X_text = df['text']
    y = df['label_num']

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X_text)
    
    print("Text converted to numerical features")
    
    return X, y, vectorizer