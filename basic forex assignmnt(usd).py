import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

# 1. Generate random USD price data
np.random.seed(42)
n_days = 500
dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days)
prices = np.cumsum(np.random.randn(n_days)) + 100  # Random walk around 100

df = pd.DataFrame({'Date': dates, 'Close': prices})
df.set_index('Date', inplace=True)

# 2. Feature engineering
df['Return'] = df['Close'].pct_change()
df['SMA_5'] = SMAIndicator(df['Close'], window=5).sma_indicator()
df['SMA_10'] = SMAIndicator(df['Close'], window=10).sma_indicator()
df['RSI_14'] = RSIIndicator(df['Close'], window=14).rsi()

# 3. Target: 1 if next day's close is higher, else 0
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# 4. Drop NaNs from indicators and target
df = df.dropna()

# 5. Features and target split
features = ['Return', 'SMA_5', 'SMA_10', 'RSI_14']
X = df[features]
y = df['Target']

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 7. Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 8. Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.2f}")

# 9. Predict up/down for the last available day
last_row = df.iloc[[-1]][features]
prediction = model.predict(last_row)[0]
print("Prediction for the next day:", "UP" if prediction == 1 else "DOWN")