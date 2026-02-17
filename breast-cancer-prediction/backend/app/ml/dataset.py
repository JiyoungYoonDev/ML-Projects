from sklearn.datasets import load_breast_cancer
import pandas as pd

def get_breast_cancer_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    n_samples, n_features = X.shape

    count0 = int((y == 0).sum())
    count1 = int((y == 1).sum())


    return {
        "dataset": "sklearn_breast_cancer",
        "n_samples": n_samples,
        "n_features": n_features,
        "target_names": data.target_names.tolist(),
        "feature_names": data.feature_names.tolist(),
        "class_distribution": {
            "0": {
                "name": data.target_names[0],
                "count": count0,
                "rate": round(count0 / n_samples, 4)
            },
            "1": {
                "name": data.target_names[1],
                "count": count1,
                "rate": round(count1 / n_samples, 4)
            }
        }
    }