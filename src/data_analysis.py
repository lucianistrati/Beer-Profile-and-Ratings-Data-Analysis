import pandas as pd

def preprocess_data(file_path):
    """
    Preprocesses beer profile and ratings data.

    Args:
    file_path (str): The file path to the CSV file.

    Returns:
    pandas.DataFrame: The preprocessed DataFrame.
    """
    df = pd.read_csv(file_path)
    print(df.head())
    print(df.describe())
    print(f"Num. records: {len(df)}")
    print(f"Style possible values: {len(set(df['Style'].to_list()))}")
    print(f"Description possible values: {len(set(df['Description'].to_list()))}")
    print(df.columns)
    print(df.head(1))
    print(df.dtypes)
    # print(df['Description'].to_list()[:10])
    # print(df['Style'].to_list()[:80])
    print(df.dtypes)
    # print(df['Description'])

    print(f"Description possible values: {len(set(df['Description'].to_list()))}")
    print(len(df))
    df = df.drop_duplicates(subset="Description")
    df.to_csv("data/beer_profile_and_ratings_no_dups.csv")
    print(f"Description possible values: {len(set(df['Description'].to_list()))}")
    print(len(df))
    print(df.dtypes)

    # dif de Max Ibu si Min IBU
    print(f"Description possible values: {set(df['Max IBU'].to_list())}")

    return df
