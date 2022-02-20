import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import typing
import umap
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, MeanShift
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
from sklearn.cluster import SpectralClustering, Birch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA_dim
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.svm import SVR, SVC
from sklearn_extra.cluster import KMedoids
from xgboost import XGBRegressor, XGBClassifier
from gensim.models import Word2Vec

taste_cols = ['Bitter', 'Sweet', 'Sour', 'Salty']
mouthfeel_cols = ['Astringency', 'Body', 'Alcohol']
flavor_aroma_cols = ['Fruits', 'Hoppy', 'Spices', 'Malty']

numerical_feature_columns = ['ABV', 'Min IBU', 'Max IBU', 'Astringency', 'Body', 'Alcohol', 'Bitter', 'Sweet', 'Sour', 'Salty',
                             'Fruits', 'Hoppy', 'Spices', 'Malty', 'review_aroma', 'review_appearance', 'review_palate',
                             'review_taste', 'review_overall', 'number_of_reviews']

beer_related_numerical_feature_columns = ['ABV', 'Min IBU', 'Max IBU', 'Astringency', 'Body', 'Alcohol', 'Bitter', 'Sweet', 'Sour', 'Salty',
                                          'Fruits', 'Hoppy', 'Spices', 'Malty']

review_related_feature_columns = ['review_aroma', 'review_appearance', 'review_palate',
                                  'review_taste', 'review_overall', 'number_of_reviews']

def document_preprocess(document):
    return word_tokenize(document)
import random


def get_metrics(clusterizer, data, labels, ground_truth):
    print("*" * 5)
    print(clusterizer," results:")
    # import pdb
    # pdb.set_trace()
    print("silhouette:", silhouette_score(data, labels))
    print("homogeneity: ", homogeneity_score(ground_truth, labels))
    print("adjusted rand score:", adjusted_rand_score(labels, ground_truth))
    print("adjusted mutual info score:", adjusted_mutual_info_score(labels, ground_truth))
    print("*" * 5)

def cluster(data, true_labels, clusterizer, max_num_datapoints: int = 1000):

    # scaler = StandardScaler()
    # data = scaler.fit_transform(data)
    print("Before clusterizing")
    clusterizer.fit_predict(data)
    print("After clusterizing")
    predicted_labels = clusterizer.labels_

    get_metrics(clusterizer=clusterizer, data=data, labels=predicted_labels, ground_truth=true_labels)


    if max_num_datapoints is not None:
        c = list(zip(data, predicted_labels, true_labels))
        random.shuffle(c)
        data, predicted_labels, true_labels = zip(*c)

        data = data[:max_num_datapoints]
        predicted_labels = predicted_labels[:max_num_datapoints]
        true_labels = true_labels[:max_num_datapoints]

    for n_components in [2 ,3]:
        for dim_red_option in ["PCA", "TSNE"]:
            if dim_red_option == "PCA":
                dim_reducer = PCA(n_components=n_components)
            elif dim_red_option == "TSNE":
                dim_reducer = TSNE(n_components=n_components)
            else:
                raise Exceptiong("wrong dim_red_option given!")
            reduced_data = dim_reducer.fit_transform(data)
            colors = {0: "b", 1: "r", 2: "g", 3: "c", 4: "m", 5: "y", 6: "k"}
            if n_components == 2:
                for (x, label) in list(zip(reduced_data, predicted_labels)):
                    plt.scatter(x[0], x[1], color=colors[label])
                plt.gca().set_aspect('equal', 'datalim')
            elif n_components == 3:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                for (x, label) in list(zip(reduced_data, predicted_labels)):
                    ax.scatter(x[0], x[1], x[2], color=colors[label])
            else:
                raise Exception(f"n_components has to be 2 or 3 while {n_components} was given!")
            plt.title(f'{dim_red_option} - {str(clusterizer)}', fontsize=24)
            plt.savefig(f'data/images/{dim_red_option} projection in {n_components}D for {str(clusterizer)}.png')
            plt.show()



import numpy as np
from src.finetune_model import finetune_model
from sklearn.naive_bayes import MultinomialNB
from src.train_word2vec import load_beer_profile_dataset

word2vec_model = Word2Vec.load("word2vec.model")
def get_embedding(text):
    try:
        vector = word2vec_model.wv[document_preprocess(text)]
        vector = np.mean(vector, axis=0)
        vector = np.reshape(vector, (1, vector.shape[0]))
        # print(vector.shape)
    except KeyError:
        vector = np.random.rand(1, 100)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!" * 80)
    return vector

def load_labels():
    return pd.read_csv("data/beer_profile_and_ratings_no_dups.csv")["ABV"].to_list()


def train_classifier(data, labels, classifier):
    data_scaler = MinMaxScaler()
    intervals = {(min(labels), 3.999): 0,
                 (4, 4.9999): 1,
                 (5, 7.4999): 2,
                 (7.5, max(labels)): 3}

    labels = np.array([intervals[(l, r)] for label in labels for (l, r) in intervals.keys() if l <= label <= r])

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

    X_train = data_scaler.fit_transform(X_train)
    X_test = data_scaler.transform(X_test)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("F1 score: ", f1_score(y_pred, y_test, average="weighted"))

def main():
    df = pd.read_csv("data/beer_profile_and_ratings_no_dups.csv")
    classifier = MultinomialNB()
    n_clusters = 4
    data = load_beer_profile_dataset()
    true_labels = load_labels()
    finetune = False
    spectral = SpectralClustering(n_clusters=n_clusters)
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    kmeans = KMeans(n_clusters=n_clusters)
    models = [spectral, hierarchical, kmeans]
    preproc_method = ["tfidf", "word2vec"][0]

    if preproc_method == "tfidf":
        # documents = [document_preprocess(document) for document in data]
        documents = data
        tf_idf_vectorizer = TfidfVectorizer(stop_words='english',
                                            max_features=20000)
        tf_idf = tf_idf_vectorizer.fit_transform(documents)
        tf_idf_norm = normalize(tf_idf)
        tf_idf_array = tf_idf_norm.toarray()

        data = tf_idf_array
    elif preproc_method == "word2vec":
        data = np.array([get_embedding(document) for document in tqdm(data)])
    # class_weight = get_class_weight(true_labels)
    if len(data.shape) == 3:
        data = np.reshape(data, (data.shape[0], data.shape[-1]))


    if finetune:
        spectral_distrib = {"eigen_solver": ["arpack", "lobpcg", "amg"], "degree": [9, 3], "gamma": [1.0, 0.1]}
        # {'degree': 3, 'eigen_solver': 'arpack', 'gamma': 0.1}
        hierarchical_distrib = {"affinity": ["l1", "l2", "euclidean"], "compute_full_tree": [ True, False, "auto"]}
        # {'affinity': 'euclidean', 'compute_full_tree': 'auto'}
        kmeans_distrib = {"max_iter": [300, 30], "algorithm":["auto", "full", "elkan"]}
        # {'algorithm': 'auto', 'max_iter': 30}
        distribs = [spectral_distrib, hierarchical_distrib, kmeans_distrib]
        for (model, distrib) in zip(models, distribs):
            if model == spectral:
                continue
            print(finetune_model(model, data, true_labels,
                           finetune_option="sk_grid_search",
                           distributions=distrib))
            print("%" * 20)
            print(model, distrib)
            print("%" * 20)
    # exit(0)



    train_classifier(data, true_labels, classifier)

    abv_target = "ABV"
    plot_hist = True
    targets = np.array(df[abv_target].to_list())
    if plot_hist:
        plt.hist(targets, bins=25)
        plt.title(f"{abv_target} histogram")
        plt.savefig(f"data/images/{abv_target}_histogram.png")
        plt.show()

    for model in models:
        cluster(data=data, true_labels=true_labels, clusterizer=model)



if __name__ == '__main__':
    main()
