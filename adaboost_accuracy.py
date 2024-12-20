# Part 2
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the digits dataset
data = load_digits()

# Select digits 1 and 8
digit1 = 1
digit8 = 8

# Filter digits 1 and 8 from the dataset
X = data.data[(data.target == digit1) | (data.target == digit8)]
y = data.target[(data.target == digit1) | (data.target == digit8)]

# Transform labels to {-1, 1}
y[y == digit1] = -1
y[y == digit8] = 1

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create AdaBoost classifier with 50 weak learners
clf = AdaBoostClassifier(n_estimators=50)

# Train the classifier
clf.fit(X_train, y_train)

# Predict on train and test sets
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Calculate accuracies
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Train Accuracy: {train_accuracy:.5f}")
print(f"Test Accuracy: {test_accuracy:.5f}")