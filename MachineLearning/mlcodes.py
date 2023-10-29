#DECISION TREE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load the CSV file into a DataFrame
data = pd.read_csv("../salaries.csv")  # Replace "jobs.csv" with your file's name
# Select the columns for features (X) and the target (y)
X = data[["company", "job", "degree"]]
y = data["salary_more_then_100k"]
# Convert categorical variables (company, job, degree) into numerical format
X = pd.get_dummies(X, drop_first=True)  # One-hot encoding for categorical variables
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a Decision Tree classifier
decision_tree_classifier = DecisionTreeClassifier()
# Train the classifier on the training data
decision_tree_classifier.fit(X_train, y_train)
# Make predictions on the testing data
y_pred = decision_tree_classifier.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
# Print classification report and confusion matrix
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test,y_pred))

********************************************************************************************************************

LINEAR REGRESSION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("Experience-Salary.csv",converters={"exp(in months)":int()})
# y = Bo + Bi X
x= data['exp(in months)']
print(x)
y = data['salary(in thousands)']
plt.scatter(x,y)
plt.show
mean_x = np.mean(x)
mean_y = np.mean(y)
print(mean_y)
numerator =0
denominatr =0
L = len(x)
for i in range(L):
    ab = (x[i]-mean_x)*(y[i]-mean_y)
    cd = (x[i]-mean_x)**2
    numerator += ab
    denominatr += cd 
ans = numerator/denominatr
print(ans)
reg = mean_y - (ans * mean_x)
max_X = np.max(x) +100
min_y = np.min(y) -100
X = np.linspace(max_X,min_y,100)
Y = reg + ans*X
plt.plot(X,Y,color='green')
plt.scatter(x,y)
plt.xlabel("op")
plt.ylabel('sal')
plt.legend()
plt.show()

********************************************************************************************************************

#LOGISTIC REGRESSION
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load the breast cancer dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = ["ID", "Diagnosis"] + [f"Feature_{i}" for i in range(1, 31)]
data = pd.read_csv(url, names=column_names)
# Split the dataset into features and target
X = data.drop(["ID", "Diagnosis"], axis=1)
y = data["Diagnosis"]
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a Logistic Regression classifier
logistic_classifier = LogisticRegression()
# Train the classifier on the training data
logistic_classifier.fit(X_train, y_train)
# Make predictions on the test data
y_pred = logistic_classifier.predict(X_test)
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
# Display classification report and confusion matrix
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


********************************************************************************************************************

#SUPPORT VECTOR MACHINES
#SVM
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load the breast cancer dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = ["ID", "Diagnosis"] + [f"Feature_{i}" for i in range(1, 31)]
data = pd.read_csv(url, names=column_names)
# Split the dataset into features and target
X = data.drop(["ID", "Diagnosis"], axis=1)
y = data["Diagnosis"]
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create an SVM classifier
svm_classifier = SVC(kernel='linear')
# Train the classifier on the training data
svm_classifier.fit(X_train, y_train)
# Make predictions on the training data
y_train_pred = svm_classifier.predict(X_train)
# Calculate the accuracy of the model on the training data
train_accuracy = accuracy_score(y_train, y_train_pred)
# Display classification report and confusion matrix for training data
print("Training Data:")
print("Accuracy:", train_accuracy)
print("Classification Report:")
print(classification_report(y_train, y_train_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_train, y_train_pred))
# Make predictions on the test data
y_test_pred = svm_classifier.predict(X_test)
# Calculate the accuracy of the model on the test data
test_accuracy = accuracy_score(y_test, y_test_pred)
# Display classification report and confusion matrix for test data
print("\nTesting Data:")
print("Accuracy:", test_accuracy)
print("Classification Report:")
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))



********************************************************************************************************************

#DATA PREPROCESSION

# importing libraries 
import numpy as np 
import pandas as pd
import matplotlib as plt 
# dataset read 
data = pd.read_csv('data.csv')
df = pd.DataFrame(data)
df
#1. Ireplace missing values
# Perform mean imputation for the 'Salary' column
mean_salary = df['Salary'].mean()
mean_age = df['Age'].mean()
df['Salary'].fillna(mean_salary, inplace=True)
df['Age'].fillna(mean_age, inplace=True)
df
# set decimal precision to 2
df['Salary'] = df['Salary'].round(2)
df['Age'] = (df['Age']).astype(int)
df

# 2. Anomaly Detection - 
new_data = {
    'Country': 'pakistan',
    'Age': 140,
    'Salary': 1000000,
    'Purchased': 'No',
}
new_row = pd.DataFrame([new_data])
# Append the new row to the original DataFrame
df = pd.concat([df, new_row], ignore_index=True)
df1 =df.copy()
# # Calculate the z-scores for the 'Age' and 'Salary' columns
df1['Age_ZScore'] = (df['Age'] - df['Age'].mean()) / df['Age'].std(ddof=0)
df1['Salary_ZScore'] = (df['Salary'] - df['Salary'].mean()) / df['Salary'].std(ddof=0)
df
# Find the tuples with the maximum absolute z-score values
max_abs_age_zscore = df1['Age_ZScore'].abs().max()
max_abs_salary_zscore = df1['Salary_ZScore'].abs().max()
# Filter the DataFrame to include only the tuples with maximum absolute z-score values
max_abs_age_tuples = df1[df1['Age_ZScore'].abs() == max_abs_age_zscore]
max_abs_salary_tuples = df1[df1['Salary_ZScore'].abs() == max_abs_salary_zscore]
max_abs_age_tuples
max_abs_salary_tuples
hence Anomaly detected for this given tuple with salary and age deviates from normal data according to z_scores
3. Standardization 
#standardization (z-score) formula
#z = (x - μ) / σ

# # Calculate the z-scores for the 'Age' and 'Salary' columns
df2 = df.copy()
df2['Age'] = (df['Age'] - df['Age'].mean()) / df['Age'].std(ddof=0)
df2['Salary'] = (df['Salary'] - df['Salary'].mean()) / df['Salary'].std(ddof=0)
df2
4. Normalization
# Normalize 'Age' and 'Salary' columns using Min-Max scaling
df3 = df.copy()
df3['Age'] = (df['Age'] - df['Age'].min()) / (df['Age'].max() - df['Age'].min())
df3['Salary'] = (df['Salary'] - df['Salary'].min()) / (df['Salary'].max() - df['Salary'].min())
df3
Encoding
df_encoded = pd.get_dummies(df, columns=['Country', 'Purchased'], drop_first=True)
df_encoded


