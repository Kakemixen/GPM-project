# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

features = pd.read_csv('advantage_matches.csv') #"loading the data"

print('The shape of our features is:', features.shape)# "We verify that the data is loaded correctly"

labels = np.array(features['Winner'])       #"we separate labels and features"
features= features.drop('Winner', axis = 1)

print('The shape of our features is:', features.shape)

feature_list = list(features.columns) 
features = np.array(features)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.6, random_state = 0)# "once the model is train we test it on 60% of the test features"

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


rf = RandomForestRegressor(n_estimators = 1000, random_state = 0) #Creating the random forest
rf.fit(train_features, train_labels)
predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)

print('Mean Absolute Error:', np.mean(errors))

