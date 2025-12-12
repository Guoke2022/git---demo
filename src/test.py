import pandas as pd

df = pd.read_parquet("../data/trajectories/origin_trajectories/shenzhen/shenzhen_2023-04-02.parquet")
print(df.head())
