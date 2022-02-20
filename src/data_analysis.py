import pandas as pd


df = pd.read_csv("data/beer_profile_and_ratings.csv")

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