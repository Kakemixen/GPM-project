import numpy as np
import pandas as pd

def get_preprocessed_matches():
    df =  pd.read_csv("data/advantage_matches.csv",   sep=",", header=0)
    df["Type Advantage"] = df["Type Advantage"].where(df["Type Advantage"] >= 1, 0).astype(int)
    df["Winner"] = df["Winner"].astype(int)
    return pd.DataFrame(df, columns=["Type Advantage", "HP advantage", "Speed advantage",  "Atk/Def ratio", "Sp.Atk/Def ratio", "Def/Atk ratio", "Sp.Def/Atk ratio", "Winner"])


def discretify_hard(df, column, split):
    df[column] = df[column].apply(lambda x: dicretify_single_number(x, split))
    return df

def dicretify_single_number(x, split):
    if x > split:
        return 3
    elif x > 0:
        return 2
    elif x > -1 * split:
        return 1
    else:
        return 0

def discretify_ratio(df, column):
    df[column] = df[column].apply(lambda x: discretify_single_ratio(x))
    return df

def discretify_single_ratio(x):
    if x > 2:
        return 3
    elif x > 1:
        return 2
    elif x > 0.5:
        return 1
    else:
        return 0

def discretify_matches():
    df = get_preprocessed_matches()

    df = discretify_hard(df, "HP advantage", 25)
    df = discretify_hard(df, "Speed advantage", 25)
    df = discretify_ratio(df, "Atk/Def ratio")
    df = discretify_ratio(df, "Sp.Atk/Def ratio")
    df = discretify_ratio(df, "Def/Atk ratio")
    df = discretify_ratio(df, "Sp.Def/Atk ratio")

    return df

if __name__ == "__main__":
    df = discretify_matches()
    df.to_csv("data/BN/BN_discrete.csv", mode="w+", index=False)

