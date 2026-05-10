from src.data.data_loader import load_data
from src.data.preprocessing import preprocess_data
from src.pipeline.train import train_model
from src.pipeline.evaluate import evaluate
from src.utils.plotting import plot_feature_importance

def run():
    df = load_data("data/spam.csv")
    
    X, y, vectorizer = preprocess_data(df)

   

    model, X_test, y_test = train_model(X, y)

    evaluate(model, X_test, y_test)

    plot_feature_importance(model,vectorizer)

if __name__ == "__main__":
    run()