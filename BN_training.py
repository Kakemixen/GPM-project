import pgmpy as pgm
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from BN import create_BN

G = create_BN()

df  = pd.read_csv("data/BN/BN_discrete.csv",   sep=",", header=0)
df_train, df_test = train_test_split(df, test_size=0.1) #500 rows for validating

#need this?
df_train_target = df_train["Winner"]
df_train_features = df_train.drop("Winner", axis=1)
df_test_target = df_test["Winner"]
df_test_features = df_test.drop("Winner", axis=1)
print("Training")
# Now that we are done with creating the model, we can train the CPD's

#proof of consept limit
n = 500

#init
prediction_change = True
new_model = None
old_model = G

limited_training_set = df_train.iloc[:n]
limited_testing_set = df_train.iloc[:(n//10)]

hidden_estimation = G.predict(limited_training_set.iloc[[0]])
print(hidden_estimation.sample(1))

#initialization of training set, need to have "complete" data
print("initializing prediction 0")
for i in range(1, n):
    prediction = G.predict(limited_training_set.iloc[[i]])
    hidden_estimation = hidden_estimation.append(prediction, ignore_index=True)
print(hidden_estimation)

while prediction_change:
    print("loop")

    prediction_change = False



    # doesnt look like works with too much data

    # GIBBS
    # use model to sample from probability values for the four hidden variables | model.predict(uncomplete df)
    # use new dataset to retrain the model  | model.fit(complete df)

