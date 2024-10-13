# Import necessary libraries
import os
import pandas as pd
import numpy as np
# Load the Iris dataset

current_directory = os.path.dirname(__file__); 
data = pd.read_csv(os.path.join(current_directory, 'iris.csv'))

# Separate features and target
X =  np.array(data.iloc[:, :-1])
y = np.array(data.iloc[: , -1])


# Split the dataset into an 80-20 training-test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

# Apply feature scaling on the training and test sets

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train= sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Print the scaled training and test sets

print(X_train)
print(X_test)