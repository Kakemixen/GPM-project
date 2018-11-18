import numpy as np
import pandas as pd

def get_preprocessed_matches():
   return pd.read_csv("data/advantage_matches.csv",   sep=",", header=0)

def normalize_data(df):
    x = df.values

def normalize_after_capping_negative(df, column, max_int):
    df[column].clip(-max_int, max_int)
    if abs(df[column].min()) > df[column].max():
        df[column]=df[column]/abs(df[column].min())
    else:
        df[column]=df[column]/df[column].max()
    return df

def normalize_non_negative(df, column):
    df[column]=df[column]/df[column].max()
    return df

def normalize_adv_matches():
    df = get_preprocessed_matches()

    df = normalize_after_capping_negative(df, "HP advantage", 100)
    df = normalize_after_capping_negative(df, "Attack advantage", 100)
    df = normalize_after_capping_negative(df, "Defense advantage", 100)
    df = normalize_after_capping_negative(df, "Sp. Atk advantage", 100)
    df = normalize_after_capping_negative(df, "Sp. Def advantage", 100)
    df = normalize_after_capping_negative(df, "Speed advantage", 100)
    df = normalize_after_capping_negative(df, "Avg. Advantage", 100)
    df = normalize_non_negative(df, "Atk/Def ratio")
    df = normalize_non_negative(df, "Sp.Atk/Def ratio")
    df = normalize_non_negative(df, "Def/Atk ratio")
    df = normalize_non_negative(df, "Sp.Def/Atk ratio")
    df["Type Advantage"] = df["Type Advantage"] -1
    df["Type Advantage"] = df["Type Advantage"].where(df["Type Advantage"] >= 0, -1)
    df = df.drop(["Unnamed: 0"], axis=1)

    return df

if __name__ == "__main__":
    df = normalize_adv_matches()
    df.to_csv("data/ANN_normalized.csv", mode="w+")

