import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv('/content/diabetes.csv')

print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())

# Features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
score = accuracy_score(y_test, predictions)

print("=" * 40)
print("Accuracy:", score)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nModel trained successfully!")

# Save model
joblib.dump(model, 'diabetes_prediction.pkl')
print("Model saved successfully!")