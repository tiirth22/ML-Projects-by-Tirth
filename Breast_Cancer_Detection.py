"""This GitHub code has been crafted by TIRTH JIGNESHKUMAR DALAL from CHARUSAT UNIVERSITY with a focus on simplifying and streamlining processes. 
It offers a user-friendly approach to solving problems, ensuring accessibility and ease of use for all stakeholders. 
The code demonstrates a commitment to clarity and efficiency, providing intuitive solutions to complex challenges. 
With clear documentation and concise implementation, this code fosters collaboration and empowers users to navigate and contribute effectively. 
It reflects a dedication to enhancing the developer experience and fostering a welcoming environment for innovation and growth."""

""" This Python code aims to distinguish between benign tumours, which are generally harmless or
mild, and malignant tumours, which are cancerous and potentially harmful. By analysing features
extracted from medical data, such as images or diagnostic tests, the code classifies tumours into 
these two categories.  
Its purpose is to aid in medical diagnosis by providing insights into the nature of tumours,  
facilitating appropriate treatment decisions and patient care."""


# DAY 1 
# Import necessary libraries 

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB                 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# DAY 2

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data  # Features
y = data.target  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0.42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit and transform the training data
X_test = scaler.transform(X_test)  # Transform the testing data using the same scaler

# Initialize and train the logistic regression model
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, y_train)  # Fit the model to the training data

# Make predictions on the testing set
y_pred = logistic_regression_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)  # Calculate the accuracy of the predictions

# Print the accuracy
print("Accuracy of classifying malignant breast cancer cases:", accuracy)


