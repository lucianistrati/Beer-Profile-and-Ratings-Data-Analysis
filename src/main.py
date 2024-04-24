import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import umap
from tqdm import tqdm
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, MeanShift, KMeans, SpectralClustering, Birch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA_dim
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from sklearn.svm import SVR, SVC
from sklearn_extra.cluster import KMedoids
from xgboost import XGBRegressor, XGBClassifier
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Ignore DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Define constants
taste_cols = ['Bitter', 'Sweet', 'Sour', 'Salty']
mouthfeel_cols = ['Astringency', 'Body', 'Alcohol']
flavor_aroma_cols = ['Fruits', 'Hoppy', 'Spices', 'Malty']
numerical_feature_columns = ['ABV', 'Min IBU', 'Max IBU', 'Astringency', 'Body', 'Alcohol', 'Bitter', 'Sweet', 'Sour', 'Salty',
                             'Fruits', 'Hoppy', 'Spices', 'Malty', 'review_aroma', 'review_appearance', 'review_palate',
                             'review_taste', 'review_overall', 'number_of_reviews']
beer_related_numerical_feature_columns = ['ABV', 'Min IBU', 'Max IBU', 'Astringency', 'Body', 'Alcohol', 'Bitter', 'Sweet', 'Sour', 'Salty',
                                          'Fruits', 'Hoppy', 'Spices', 'Malty']
review_related_feature_columns = ['review_aroma', 'review_appearance', 'review_palate', 'review_taste', 'review_overall', 'number_of_reviews']

# Preprocessing functions
def document_preprocess(document):
    """
    Preprocesses a document by tokenizing it.

    Args:
    document (str): The input document.

    Returns:
    list: A list of preprocessed tokens.
    """
    return word_tokenize(document)

def get_embedding(text):
    """
    Gets the Word2Vec embedding for a text.

    Args:
    text (str): The input text.

    Returns:
    numpy.ndarray: The Word2Vec embedding.
    """
    try:
        vector = word2vec_model.wv[document_preprocess(text)]
        vector = np.mean(vector, axis=0)
    except KeyError:
        vector = np.random.rand(1, 100)
    return vector

def load_labels(file_path):
    """
    Loads labels from a CSV file.

    Args:
    file_path (str): The path to the CSV file.

    Returns:
    list: A list of labels.
    """
    return pd.read_csv(file_path)["ABV"].tolist()

# Clustering and Evaluation functions
def get_metrics(clusterizer, data, labels, ground_truth):
    """
    Calculates clustering evaluation metrics.

    Args:
    clusterizer: The clustering model.
    data (numpy.ndarray): The input data.
    labels (numpy.ndarray): The predicted labels.
    ground_truth (list): The ground truth labels.

    Returns:
    None
    """
    print("*" * 5)
    print(clusterizer, " results:")
    print("silhouette:", silhouette_score(data, labels))
    print("homogeneity: ", homogeneity_score(ground_truth, labels))
    print("adjusted rand score:", adjusted_rand_score(labels, ground_truth))
    print("adjusted mutual info score:", adjusted_mutual_info_score(labels, ground_truth))
    print("*" * 5)

def cluster(data, true_labels, clusterizer, max_num_datapoints=None):
    """
    Performs clustering and visualization.

    Args:
    data (numpy.ndarray): The input data.
    true_labels (list): The true labels.
    clusterizer: The clustering model.
    max_num_datapoints (int): Maximum number of data points to consider.

    Returns:
    None
    """
    clusterizer.fit_predict(data)
    predicted_labels = clusterizer.labels_
    get_metrics(clusterizer=clusterizer, data=data, labels=predicted_labels, ground_truth=true_labels)
    if max_num_datapoints is not None:
        c = list(zip(data, predicted_labels, true_labels))
        random.shuffle(c)
        data, predicted_labels, true_labels = zip(*c)
        data = data[:max_num_datapoints]
        predicted_labels = predicted_labels[:max_num_datapoints]
        true_labels = true_labels[:max_num_datapoints]

    for n_components in [2, 3]:
        for dim_red_option in ["PCA", "TSNE"]:
            if dim_red_option == "PCA":
                dim_reducer = PCA(n_components=n_components)
            elif dim_red_option == "TSNE":
                dim_reducer = TSNE(n_components=n_components)
            else:
                raise Exception("Wrong dim_red_option given!")
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

# Main function
def main():
    # Load beer profile dataset
    df = pd.read_csv("data/beer_profile_and_ratings_no_dups.csv")

    # Define classifier
    classifier = MultinomialNB()

    # Define clustering models
    n_clusters = 4
    spectral = SpectralClustering(n_clusters=n_clusters)
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    kmeans = KMeans(n_clusters=n_clusters)
    models = [spectral, hierarchical, kmeans]

    # Load data and labels
    data = load_beer_profile_dataset()
    true_labels = load_labels("data/beer_profile_and_ratings_no_dups.csv")

    # Preprocess data
    preproc_method = "tfidf"
    if preproc_method == "tfidf":
        tf_idf_vectorizer = TfidfVectorizer(stop_words='english', max_features=20000)
        tf_idf = tf_idf_vectorizer.fit_transform(data)
        tf_idf_norm = normalize(tf_idf)
        data = tf_idf_norm.toarray()
    elif preproc_method == "word2vec":
        data = np.array([get_embedding(document) for document in tqdm(data)])
    if len(data.shape) == 3:
        data = np.reshape(data, (data.shape[0], data.shape[-1]))

    # Fine-tune clustering models
    finetune = False
    if finetune:
        for (model, distrib) in zip(models, distribs):
            if model == spectral:
                continue
            print(finetune_model(model, data, true_labels, finetune_option="sk_grid_search", distributions=distrib))
            print("%" * 20)
            print(model, distrib)
            print("%" * 20)

    # Train classifier
    train_classifier(data, true_labels, classifier)

    # Plot histogram
    abv_target = "ABV"
    plot_hist = True
    targets = np.array(df[abv_target].to_list())
    if plot_hist:
        plt.hist(targets, bins=25)
        plt.title(f"{abv_target} histogram")
        plt.savefig(f"data/images/{abv_target}_histogram.png")
        plt.show()

    # Perform clustering and visualization
    for model in models:
        cluster(data=data, true_labels=true_labels, clusterizer=model)

if __name__ == '__main__':
    main()
