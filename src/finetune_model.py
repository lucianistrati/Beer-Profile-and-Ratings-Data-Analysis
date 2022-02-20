import keras_tuner as kt
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

def silhouette_score(estimator, X):
    clusters = estimator.fit_predict(X)
    score = metrics.silhouette_score(X, clusters, metric='precomputed')
    return score

def finetune_model(model, data, targets, finetune_option="sk_grid_search", distributions: dict = dict()):
    if finetune_option == "sk_grid_search":
        finetuner = GridSearchCV(model, distributions, scoring=silhouette_score)
    search = finetuner.fit(data, targets)
    return search.best_params_

