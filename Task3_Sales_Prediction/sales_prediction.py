import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import os

warnings.filterwarnings('ignore')

# --- 1. Load and inspect the dataset ---
def load_and_inspect_data(file_path):
    """
    Loads the dataset from a CSV file and performs initial inspection.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        print("Please ensure the 'Advertising.csv' file is in the correct directory.")
        return None

    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully.")

        # Drop the 'Unnamed: 0' column if it exists, as seen in many versions of this dataset.
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
            print("\nDropped 'Unnamed: 0' column.")
            
        print("\n--- First 5 rows of the dataset ---")
        print(df.head())
        print("\n--- Dataset Info ---")
        df.info()
        print("\n--- Missing values per column ---")
        print(df.isnull().sum())

        return df
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None

# --- 2. Exploratory Data Analysis (EDA) ---
def perform_eda(df):
    """
    Performs basic exploratory data analysis with visualizations.
    """
    print("\n--- Statistical Summary ---")
    print(df.describe())

    # Visualize the relationship between features and sales
    print("\n--- Creating pairplot for visualization... ---")
    sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', kind='reg', height=4, aspect=0.7)
    plt.suptitle('Advertising Spend vs. Sales', y=1.02)
    plt.show()

    # Create a heatmap for correlation
    print("\n--- Creating correlation heatmap... ---")
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Advertising and Sales')
    plt.show()

# --- 3. Data Preprocessing and Splitting ---
def prepare_data(df):
    """
    Splits the data into features (X) and target (y), then into training and testing sets.
    """
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n--- Data splitting complete ---")
    print(f"Training features shape: {X_train.shape}")
    print(f"Testing features shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test

# --- 4. Model Training ---
def train_model(X_train, y_train):
    """
    Initializes and trains a Linear Regression model.
    """
    print("\n--- Training Linear Regression model... ---")
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Model trained successfully.")
    print(f"Model Intercept: {model.intercept_}")
    print(f"Model Coefficients (TV, Radio, Newspaper): {model.coef_}")

    return model

# --- 5. Model Evaluation ---
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model using test data and visualizes the results.
    """
    print("\n--- Evaluating model performance ---")
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")

    # Plotting actual vs. predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title('Actual vs. Predicted Sales')
    plt.grid(True)
    plt.show()

# --- 6. Prediction on New Data ---
def predict_new_sales(model):
    """
    Predicts sales for a new set of advertising budgets.
    """
    print("\n--- Making predictions for new data ---")
    # You can change these values to test different scenarios
    new_budgets = pd.DataFrame({
        'TV': [150, 200],
        'Radio': [20, 40],
        'Newspaper': [10, 50]
    })

    predicted_sales = model.predict(new_budgets)

    print("Predicted sales for new advertising budgets:")
    for i, sales in enumerate(predicted_sales):
        print(f"  - Budget set {i+1} (TV: {new_budgets['TV'][i]}, Radio: {new_budgets['Radio'][i]}, Newspaper: {new_budgets['Newspaper'][i]}): Predicted Sales = {sales:.2f}")

# --- Main Execution Flow ---
if __name__ == "__main__":
    file_name = "Advertising.csv"

    advertising_df = load_and_inspect_data(file_name)
    if advertising_df is None:
        exit()

    perform_eda(advertising_df)
    X_train, X_test, y_train, y_test = prepare_data(advertising_df)
    linear_model = train_model(X_train, y_train)
    evaluate_model(linear_model, X_test, y_test)
    predict_new_sales(linear_model)

