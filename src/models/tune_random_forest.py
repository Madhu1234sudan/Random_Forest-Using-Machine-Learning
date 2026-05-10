from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def tune_model(X_train, y_train):

    param_grid = {
        'n_estimators' : [100, 200],
        'max_depth' : [None, 20],
        'main_samples_split' : [2, 5]
    }

    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator= rf,
        param_grid= param_grid,
        cv= 3,
        scoring= 'f1',
        n_jobs= -1
    )

    grid_search.fit(X_train, y_train)

    print("Best parameters: ", grid_search.best_params_)
    print("Best Score: ", grid_search.best_score_)


    return grid_search.best_estimator_