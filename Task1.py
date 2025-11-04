import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# sample dataset
data = {
    'square_feet': [1500, 1800, 2400, 3000, 3500, 4000, 1300, 2200, 2800, 3600],
    'bedrooms': [3, 4, 3, 5, 4, 5, 2, 3, 4, 5],
    'bathrooms': [2, 3, 2, 4, 3, 4, 1, 2, 3, 4],
    'price': [320000, 400000, 430000, 550000, 620000, 700000, 280000, 410000, 520000, 640000]
}

df = pd.DataFrame(data)

# Define features (X) and target (y)
X = df[['square_feet', 'bedrooms', 'bathrooms']]
y = df['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Display model coefficients
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")

print("\nIntercept:", model.intercept_)

# Example prediction
new_house = pd.DataFrame({
    'square_feet': [2500],
    'bedrooms': [4],
    'bathrooms': [3]
})

predicted_price = model.predict(new_house)
print("\nPredicted price for the new house:", predicted_price[0])


# Compare actual vs predicted prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')

plt.title('Actual vs Predicted House Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.grid(True)
plt.show()