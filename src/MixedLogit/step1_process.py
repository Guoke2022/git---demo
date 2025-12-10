import pandas as pd
import numpy as np

def process_data(data_path:str,out_path:str):
    
    """
    Process data
    """

    # load data
    df = pd.read_parquet(data_path)

    print("load data")
    print(df.describe())

    # ID encoding
    df["id"] = df["id"].astype("category").cat.codes.astype("int32")

    # Missing value handling
    ids_with_missing = df[df.isnull().any(axis=1)]["id"].unique()
    df = df[~df["id"].isin(ids_with_missing)]

    print("Missing value handling")

    # SES reclassification
    df["SES_Level"] = pd.qcut(df["SES"],q=[0,0.2,0.8,1],labels=["Low","Middle","High"])

    print("SES reclassification")  

    # Label Encoding
    df["option_grade"] = df["option_grade"].map({
                            "0":0,
                            "Primary":1,
                            "Secondary":2,
                            "Tertiary":3,
                            "Tertiary grade A":4}).astype("int8",copy=False)

    print("Label Encoding") 

    # Unit conversion
    df["option_distance_100_km"] = (df["option_distance_m"]/100_000).astype("float16")
    df["option_beds"] = (df["option_beds"]/100).astype("float16",copy=False)

    print("Unit conversion") 

    # Add quadratic and cubic terms for distance
    df["option_distance_100_km_2"] = ((df["option_distance_100_km"]**2)).astype("float16")
    df["option_distance_100_km_3"] = ((df["option_distance_100_km"]**3)).astype("float16")

    print("Add quadratic and cubic terms for distance.")

    # Clear unnecessary terms
    df.drop(columns=["option_distance_m","SES"],inplace=True)

    # Sorting
    df.sort_values(by=["id","option_hosp_id"],inplace=True)
    print("Sorting") 

    # save data
    df.to_parquet(out_path)

if __name__=="__main__":
    data_path = "data/raw/AllCities_id_hosp_distance_with_features.parquet"
    out_path = "data/process/all_city.parquet"
    process_data(data_path,out_path)
