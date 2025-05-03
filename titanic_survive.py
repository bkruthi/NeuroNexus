# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the Titanic dataset
url = 'C:\\Users\\jtote\\Downloads\\archive (5)\\tested.csv'
titanic_data = pd.read_csv(url)

# Preprocess the data (Handle missing values, encode categorical variables)
# Drop columns that won't be useful for the model
titanic_data = titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# Fill missing values (if any) using forward fill
titanic_data.fillna(method='ffill', inplace=True)

# Encode categorical columns (Sex and Embarked)
le_sex = LabelEncoder()
titanic_data['Sex'] = le_sex.fit_transform(titanic_data['Sex'])

le_embarked = LabelEncoder()
titanic_data['Embarked'] = le_embarked.fit_transform(titanic_data['Embarked'])

# Define features (X) and target (y)
X = titanic_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = titanic_data['Survived']

# Initialize the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Apply 5-Fold Cross-Validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

# Print cross-validation results
print("Cross-validation scores:", cv_scores)
print("Average cross-validation score:", cv_scores.mean())

# To check how the model works on test data, split the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model on the training set
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate model performance on test set
test_accuracy = accuracy_score(y_test, y_pred)
print("Test set accuracy:", test_accuracy)

# Prediction for new test data (you can add a new row of data here)
new_data = pd.DataFrame({
    'Pclass': [3],
    'Sex': le_sex.transform(['female']),
    'Age': [22],
    'SibSp': [1],
    'Parch': [0],
    'Fare': [7.25],
    'Embarked': le_embarked.transform(['S'])
})

# Predict survival for the new data
new_prediction = model.predict(new_data)
prediction_result = 'Survived' if new_prediction[0] == 1 else 'Did not survive'
print("Prediction for the new data:", prediction_result)
