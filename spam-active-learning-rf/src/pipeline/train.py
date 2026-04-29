from src.models.random_forest import get_model

def train_model(X, y):
    model = get_model()
    model.fit(X, y)
    print("Model Trained Successfully")
    return model