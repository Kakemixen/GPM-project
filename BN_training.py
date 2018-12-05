import sys
import time
from  os import get_terminal_size
import pgmpy as pgm
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import BN

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
        print("predicting winner: {:<5} of {}".format(i+1, n), end="\r")
        prediction = G.predict(features.iloc[[i]])
        predicted_winner = prediction["Winner"]
        if predicted_winner.iloc[0] == winners.iloc[i]:
            correct += 1
        else:# TODO confidence (interesting if this is uncertain, P ~ 0.5)
            pass
    print()
    return correct / n

# hidden_estimation = G.predict(limited_training_set)
def predict_hidden(G, data):
    n = len(data.index)
    hidden_estimation = G.predict(data.iloc[[0]])
    for i in range(1, n):
        print("estimating row: {:<5} of {}".format(i+1, n), end="\r")
        prediction = G.predict(data.iloc[[i]])
        hidden_estimation = hidden_estimation.append(prediction, ignore_index=True)
    print()
    return combine(hidden_estimation, data)

def train_BN(domain=True, many=False):
    start_time = time.time()
    #proof of consept limit
    n = 100
    margin = 0.01

    df  = pd.read_csv("data/BN/BN_discrete.csv",   sep=",", header=0)
    df_train, df_test = train_test_split(df, test_size=0.1) #5000 rows for validating


    #init
    prediction_change = True
    old_accuracy = 0
    limited_training_set = df_train.iloc[:n]
    limited_valid_set = df_train.iloc[:(n//5)]
    w, h = get_terminal_size()

    ####Get model + initialize latent variables
    # uses initialized BN
    if domain:
        G = BN.initialize_G_4(BN.create_BN_4()) if many else BN.initialize_G_4(BN.create_BN_2())
        print("initial predict")
        estimated_data = predict_hidden(G, limited_training_set)
    else:
        # # uses non-initializedBN
        G = BN.create_BN_4()
        # find hidden variables (nodes) in G
        latent_variables = [ x for x in G.nodes() if x not in df.columns]
        # estimated_data = random values with same name as the hidden variabels in G
        estimated_data = pd.DataFrame(limited_training_set.copy())
        for hidden in latent_variables:
            estimated_data[hidden] = np.random.randint(0, 3, estimated_data.shape[0])

    max_iter = 500
    accuracy_avg = 0
    last_accuracies = [0]
    accuracies = []
    n_last = 3
    for i in range(max_iter): #set max iterations to some number
        print("iteration: {}".format(i))
        # doesnt look like works with too much data

        #train the model with the estimated data
        G.fit(estimated_data)

        accuracy = calc_accuracy(G, limited_training_set)
        # NOTE: drop in accuracy at some point, which is wierd
            # on that note, see if we can use predict_probability() instead?
            # further on that note, seems like close to convergence at 2nd epoch (especially for initialized)
            # could try to get metrics per initialization (random columns, unbiased columns, biased columns, initialized  BN...)

        # quickfix | checks agains average over n_last iterations
        if(len(last_accuracies) >= n_last): last_accuracies.pop(0)
        last_accuracies.append(accuracy)
        accuracies.append(accuracy)
        accuracy_avg = sum(last_accuracies) / len(last_accuracies)

        print("accuracy: {} vs avg: {}".format(accuracy, accuracy_avg))
        if accuracy < accuracy_avg + margin:break

        """
        if accuracy < old_accuracy + margin:break
        old_accuracy = accuracy
        """


        #predict new hidden CPT's
        estimated_data = predict_hidden(G, limited_training_set)

        # GIBBS
        # use model to sample from probability values for the four hidden variables | model.predict(uncomplete df)
        # use new dataset to retrain the model  | model.fit(complete df)

        # #1 full iteration
        # for z in range(n):
        #     print("iteration: {:<4} - Gibbs: {:<4}".format(i, z), end="\r")
        #     estimated_data.iloc[z] = row_estimate(G, z, estimated_data, latent_columns)
    print("done trainin, testing")
    test_acc = calc_accuracy(G, df_test[:2*n])
    print("test accuracy: {}".format(test_acc))

    finish_time = time.time()
    #writing run-data to file
    with open("data/BN/accuracies_{}_n={}.txt".format("many" if many else "few", n), "a+") as f:
        # format line_number => run_num | accuracies** delim="," last element test_acc
        for acc in accuracies:
            f.write(str(acc) + ",")
        f.write(test_acc)
        f.write("\n")
    with open("data/BN/times_{}_n={}.txt".format("many" if many else "few", n), "a+") as f:
        # format line_number => run_num | time as time.time()
        f.write("{}\n".format(finish_time - start_time))
    print("appended run-accuracies to data/BN/accuracies_{}_n={}.txt".format("many" if many else "few", n))
    print("appended run-time to data/BN/times_{}_n={}.txt".format("many" if many else "few", n))
def main():

    if len(sys.argv) > 1 and sys.argv[1] == "1":
        print("many")
        train_BN(domain=False, many=True)
    else:
        print("few")
        train_BN(domain=False, many=False)

if __name__ == "__main__":
    main()
