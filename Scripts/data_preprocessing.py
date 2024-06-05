import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load your dataset using pandas
data = pd.read_csv('../data/credit_card_churn.csv')

# Handle missing values
# Replace missing values with the mean of each column
data.fillna(data.mean(), inplace=True)

# Encode categorical variables (if applicable)
data = pd.get_dummies(data)

# Identify the numerical feature columns
numerical_features = ['Transaction Amount', 'Transaction Time', 'Customer Age', 'Account Age']

# Scale numerical features
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Split the data into training and testing sets
# 'Fraudulent' is the target variable
X = data.drop('Fraudulent', axis=1)
y = data['Fraudulent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the processed data to CSV files
X_train.to_csv('../data/X_train.csv', index=False)
X_test.to_csv('../data/X_test.csv', index=False)
y_train.to_csv('../data/y_train.csv', index=False)
y_test.to_csv('../data/y_test.csv', index=False)

print("Data preprocessing completed successfully.")
