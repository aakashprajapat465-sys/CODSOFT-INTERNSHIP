
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#  Load the dataset
df = pd.read_csv("Titanic-Dataset.csv")

print("Preview of the dataset:")
print(df.head())

print("\nBasic Info of dataset:")
print(df.info())


#  Handle missing values
age_imputer = SimpleImputer(strategy='median')
df['Age'] = age_imputer.fit_transform(df[['Age']])

# Embarked → fill with the most frequent value (mode)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop columns that don’t add much predictive power
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)


# Encode categorical features

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])


#  Split into train & test sets
X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#  Build and train the model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)


# Evaluate the model
y_pred = rf_model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Died", "Survived"],
    yticklabels=["Died", "Survived"]
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Step 9: Feature Importance
feature_importances = rf_model.feature_importances_
features = X.columns

plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importances, y=features, palette="viridis")
plt.title("Feature Importance in Titanic Survival Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()
