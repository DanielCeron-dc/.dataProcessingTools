# Importing the necessary libraries
import os
import numpy as np
import pandas as pd
# Load the dataset

current_directory = os.path.dirname(__file__)
dataset = pd.read_csv(os.path.join(current_directory, 'data.csv'))
# Identify the categorical data
categorical_features = ['Sex', 'Embarked', 'Pclass']

# Implement an instance of the ColumnTransformer class
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_features)], remainder='passthrough')

# Apply the fit_transform method on the instance of ColumnTransformer
# Convert the output into a NumPy array
X = np.array(ct.fit_transform(dataset))

# Use LabelEncoder to encode binary categorical data

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
Y = le.fit_transform(dataset['Survived']) # actually this is no needed, because the column is already binary


# Print the updated matrix of features and the dependent variable vector
print(X)
print(Y)