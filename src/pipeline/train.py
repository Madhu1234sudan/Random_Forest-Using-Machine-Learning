from sklearn.model_selection import train_test_split
from src.models.tune_random_forest import tune_model

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
    )

    model = tune_model(X_train, y_train)

    print("Best Tuned Model Trained Successfully")

    return model, X_test, y_test