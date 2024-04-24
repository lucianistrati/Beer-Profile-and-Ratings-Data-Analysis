import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
import multiprocessing

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load stopwords and initialize lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def document_preprocess(document):
    """
    Preprocesses a document by tokenizing and lemmatizing it.

    Args:
    document (str): The input document.

    Returns:
    list: A list of preprocessed tokens.
    """
    tokens = word_tokenize(document)
    preprocessed_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words]
    return preprocessed_tokens

def load_beer_profile_dataset(file_path):
    """
    Loads the beer profile dataset from a CSV file.

    Args:
    file_path (str): The path to the CSV file.

    Returns:
    list: A list of beer descriptions.
    """
    df = pd.read_csv(file_path)
    return df['Description'].tolist()

def train_word2vec_model(texts, vector_size=100, window=5, min_count=1, workers=multiprocessing.cpu_count()):
    """
    Trains a Word2Vec model on the provided texts.

    Args:
    texts (list): A list of preprocessed texts.
    vector_size (int): Dimensionality of the word vectors.
    window (int): Maximum distance between the current and predicted word within a sentence.
    min_count (int): Ignores all words with a total frequency lower than this.
    workers (int): Number of worker threads to train the model.

    Returns:
    Word2Vec: The trained Word2Vec model.
    """
    model = Word2Vec(sentences=texts, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model

if __name__ == "__main__":
    # Load beer profile dataset
    beer_profile_file = "data/beer_profile_and_ratings_no_dups.csv"
    texts = load_beer_profile_dataset(beer_profile_file)

    # Preprocess texts
    preprocessed_texts = [document_preprocess(text) for text in texts]

    # Train Word2Vec model
    model = train_word2vec_model(preprocessed_texts)

    # Save trained model
    model.save("word2vec.model")
    print("Model saved successfully.")
