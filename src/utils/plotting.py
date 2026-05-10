import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance(model,vectorizer):
    feature_names = vectorizer.get_feature_names_out()

    importances = model.feature_importances_

    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })

    top_features = feature_importance_df.sort_values(
        by = 'importance',
        ascending=False
    ).head(20)

    print("\nTop Important Features:\n")
    print(top_features)

    plt.figure(figsize=(10,6))

    plt.barh(
        top_features['feature'],
        top_features['importance']
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 20 Important Features")


    plt.gca().invert_xaxis()

    plt.tight_layout()

    plt.savefig("outputs/figures/feature_importance.png")

    plt.show()

    print("\nFeature importance plot saved.")