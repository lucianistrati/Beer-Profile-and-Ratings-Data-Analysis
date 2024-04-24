import keras_tuner as kt
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import metrics

def silhouette_score(estimator, X):
    """
    Calculates the silhouette score.

    Args:
    estimator: The clustering estimator.
    X (array-like): The input data.

    Returns:
    float: The silhouette score.
    """
    clusters = estimator.fit_predict(X)
    score = metrics.silhouette_score(X, clusters, metric='precomputed')
    return score

def finetune_model(model, data, targets, finetune_option="sk_grid_search", distributions: dict = dict()):
    """
    Fine-tunes a model using grid search or randomized search.

    Args:
    model: The model to be fine-tuned.
    data (array-like): The input data.
    targets (array-like): The target values.
    finetune_option (str): The fine-tuning option. Default is "sk_grid_search".
    distributions (dict): The hyperparameter distributions for fine-tuning.

    Returns:
    dict: The best hyperparameters.
    """
    if finetune_option == "sk_grid_search":
        finetuner = GridSearchCV(model, distributions, scoring=silhouette_score)
    else:
        finetuner = RandomizedSearchCV(model, distributions, scoring=silhouette_score)
    search = finetuner.fit(data, targets)
    return search.best_params_
