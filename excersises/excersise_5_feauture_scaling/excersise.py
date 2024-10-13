# Import necessary libraries
import os
import pandas as pd
import numpy as np

# Load the Wine Quality Red dataset
current_directory = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(current_directory, 'winequality.csv'), delimiter=';')

print(data)
print("Columns in data:", data.columns)
# Separate features and target
X = np.array(data.iloc[:, :-1])
y = np.array(data.iloc[:, -1])

# Split the dataset into an 80-20 training-test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create an instance of the StandardScaler class
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# Fit the StandardScaler on the features from the training set and transform it
X_train =  sc.fit_transform(X_train)
# Apply the transform to the test set
X_test = sc.transform(X_test)


# Print the scaled training and test datasets
print(X_train)
print(X_test)