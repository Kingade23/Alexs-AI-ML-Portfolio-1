# Currency Exchange Rate Predictor
# --------------------------------
# This project uses simple linear regression to predict
# the future value of an exchange rate (USD to NGN)
# based on past daily exchange rates.

# ---- STEP 1: Import necessary libraries ----
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---- STEP 2: Create or load sample financial data ----
# (You can later replace this with real data from an API or CSV)
data = {
    'Day': [1, 2, 3, 4, 5, 6, 7],
    'Rate': [1500, 1510, 1505, 1520, 1515, 1530, 1525]  # Example USD→NGN rates
}
df = pd.DataFrame(data)

# ---- STEP 3: Split the data into features (X) and target (y) ----
X = df[['Day']]     # Independent variable
y = df['Rate']      # Dependent variable

# ---- STEP 4: Create and train the regression model ----
model = LinearRegression()
model.fit(X, y)

# ---- STEP 5: Predict the next day's exchange rate ----
next_day = [[8]]  # You can change this value
predicted_rate = model.predict(next_day)

print(f"Predicted Exchange Rate for Day {next_day[0][0]}: ₦{predicted_rate[0]:.2f}")

# ---- STEP 6: Visualize the trend ----
plt.plot(df['Day'], df['Rate'], 'bo-', label='Actual Rates')
plt.plot(next_day, predicted_rate, 'ro', label='Predicted Rate')
plt.xlabel('Day')
plt.ylabel('Exchange Rate (₦)')
plt.title('USD/NGN Exchange Rate Prediction')
plt.legend()
plt.grid(True)
plt.show()
