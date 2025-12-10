
# 导入相关的包
import pandas as pd
import numpy as np

def process_data(data_path:str,out_path:str):
    
    """
    处理数据
    """

    # 读取数据
    df = pd.read_parquet(data_path)

    print("完成数据读取")
    print(df.describe())

    # id编码减少内存占用
    df["id"] = df["id"].astype("category").cat.codes.astype("int32")

    # 缺失值处理
    ids_with_missing = df[df.isnull().any(axis=1)]["id"].unique()
    df = df[~df["id"].isin(ids_with_missing)]

    print("完成缺失值处理")

    # SES重分组
    df["SES_Level"] = pd.qcut(df["SES"],q=[0,0.2,0.8,1],labels=["Low","Middle","High"])

    print("完成分组处理")  

    # Label Encoding编码
    df["option_grade"] = df["option_grade"].map({
                            "0":0,
                            "Primary":1,
                            "Secondary":2,
                            "Tertiary":3,
                            "Tertiary grade A":4}).astype("int8",copy=False)

    print("完成Label Encoding编码处理") 

    # 单位变换
    df["option_distance_100_km"] = (df["option_distance_m"]/100_000).astype("float16")
    df["option_beds"] = (df["option_beds"]/100).astype("float16",copy=False)

    print("完成单位变换处理") 

    # 添加距离二次项、三次项
    df["option_distance_100_km_2"] = ((df["option_distance_100_km"]**2)).astype("float16")
    df["option_distance_100_km_3"] = ((df["option_distance_100_km"]**3)).astype("float16")

    print("完成距离二次项添加处理")

    # 清楚无用的项
    df.drop(columns=["option_distance_m","SES"],inplace=True)

    # id和医院id排序
    df.sort_values(by=["id","option_hosp_id"],inplace=True)
    print("id排序") 

    # 保存数据
    df.to_parquet(out_path)

if __name__=="__main__":
    data_path = "data/raw/AllCities_id_hosp_distance_with_features.parquet"
    out_path = "data/process/all_city.parquet"
    process_data(data_path,out_path)