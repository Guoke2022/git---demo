import pandas as pd

df = pd.read_parquet("../data/trajectories/origin_trajectories/shenzhen/shenzhen_2023-04-02.parquet",
    engine="fastparquet")
print(df.dtypes)
print(df.head())
