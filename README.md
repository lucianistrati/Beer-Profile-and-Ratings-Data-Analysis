## Beer Profile and Ratings Data Analysis

This repository contains code for analyzing beer profile and ratings data. It includes various scripts for data preprocessing, model tuning, and visualization.

### Files in the Repository:

1. **data/beer_profile_and_ratings.csv**: This CSV file contains the raw beer profile and ratings data.

2. **preprocess_data.py**: This script preprocesses the raw data, removing duplicate entries based on the description column, and saves the cleaned data to a new CSV file named "beer_profile_and_ratings_no_dups.csv".

3. **model_tuning.py**: This script contains functions for fine-tuning machine learning models using grid search or randomized search cross-validation.

4. **data_analysis.py**: This script performs exploratory data analysis on the cleaned beer profile and ratings data, including summary statistics, value counts, and data visualization.

### Instructions:

1. **Preprocessing the Data**: Before running any analysis, ensure that you have the raw data file "beer_profile_and_ratings.csv" in the "data" directory. Run the `preprocess_data.py` script to clean the data and remove duplicate entries.

    ```bash
    python preprocess_data.py
    ```

2. **Fine-Tuning Models**: Use the `model_tuning.py` script to fine-tune machine learning models for the beer profile and ratings data. This script provides options for grid search or randomized search cross-validation.

3. **Exploratory Data Analysis**: Utilize the `data_analysis.py` script to perform exploratory data analysis on the cleaned beer profile and ratings data. This includes generating summary statistics, exploring unique values, and creating visualizations.

### Dependencies:

- pandas
- scikit-learn
- matplotlib
- numpy
- keras-tuner
- nltk (for silhouette score function)
- umap
- tqdm
- gensim

### License:

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


[![Stand With Ukraine](https://raw.githubusercontent.com/vshymanskyy/StandWithUkraine/main/banner2-direct.svg)](https://stand-with-ukraine.pp.ua)
