# Iris Flower Classification - Codsoft Internship

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# load dataset
data = pd.read_csv("IRIS.csv")
print("Dataset sample:\n", data.head())

# features and labels
X = data.drop("species", axis=1)
y = data["species"]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# logistic regression model
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)

# predictions
y_pred = clf.predict(X_test)

# evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="coolwarm",
            xticklabels=clf.classes_,
            yticklabels=clf.classes_)
plt.title("Confusion Matrix")
plt.show()

# try with one example
example = [[5.1, 3.5, 1.4, 0.2]]
print("Example:", example, "-> Prediction:", clf.predict(example)[0])
