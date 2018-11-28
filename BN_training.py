from  os import get_terminal_size
import pgmpy as pgm
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from BN import create_BN

def row_estimate(G, index, estimated_data, latent_columns):
    row = estimated_data.iloc[index]
    row = row.drop(latent_columns)

def combine(A, B):
    "combines two dataframes with no common column"
    for column in B.columns:
        A[column] = B[column].values
    return A

def calc_accuracy(G, data):
    winners = data["Winner"]
    features = data.drop("Winner", axis=1)
    n = len(data.index)
    correct = 0
    for i in range(1, n):
        print("predicting winner: {:<5} of {}".format(i, n), end="\r")
        prediction = G.predict(features.iloc[[i]])
        predicted_winner = prediction["Winner"]
        if predicted_winner.iloc[0] == winners.iloc[i]:
            correct += 1
    return correct / n

# hidden_estimation = G.predict(limited_training_set)
def predict_hidden(G, data):
    n = len(data.index)
    hidden_estimation = G.predict(data.iloc[[0]])
    for i in range(1, n):
        print("estimating row: {:<5} of {}".format(i, n), end="\r")
        prediction = G.predict(data.iloc[[i]])
        hidden_estimation = hidden_estimation.append(prediction, ignore_index=True)
    return combine(hidden_estimation, data)

def main():
    G = create_BN()

    df  = pd.read_csv("data/BN/BN_discrete.csv",   sep=",", header=0)
    df_train, df_test = train_test_split(df, test_size=0.1) #500 rows for validating

    #proof of consept limit
    n = 100

    #init
    prediction_change = True

    old_accuracy = 0

    limited_training_set = df_train.iloc[:n]
    limited_testing_set = df_test.iloc[:(n//10)]


    # latent_columns = hidden_estimation.columns

    w, h = get_terminal_size()
    i = 0
    margin = 0.001
    while prediction_change:
        print("iteration: {}".format(i))
        if i > 5000: break
        # prediction_change = False

        # doesnt look like works with too much data
        estimated_data = predict_hidden(G, limited_training_set)
        G.fit(estimated_data)
        accuracy = calc_accuracy(G, limited_testing_set)
        print("accuracy: {}".format(accuracy))
        if accuracy < old_accuracy + margin:break
        old_accuracy = accuracy

        # GIBBS
        # use model to sample from probability values for the four hidden variables | model.predict(uncomplete df)
        # use new dataset to retrain the model  | model.fit(complete df)

        # #1 full iteration
        # for z in range(n):
        #     print("iteration: {:<4} - Gibbs: {:<4}".format(i, z), end="\r")
        #     estimated_data.iloc[z] = row_estimate(G, z, estimated_data, latent_columns)

        i += 1
    print()

if __name__ == "__main__":
    main()
