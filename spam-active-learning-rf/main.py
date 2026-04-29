from src.data.data_loader import load_data
from src.data.preprocessing import preprocess_data
from src.pipeline.train import train_model
from src.pipeline.evaluate import evaluate

def run():
    df = load_data("data/spam.csv")
    
    X, y = preprocess_data(df)

   

    model = train_model(X, y)
    evaluate(model, X, y)

if __name__ == "__main__":
    run()