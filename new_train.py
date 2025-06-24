import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib

# === Step 1: Load and Clean Data ===
print("📥 Loading data...")
df = pd.read_csv("train.csv", low_memory=False).ffill()

# Keep essential columns
df = df[['Age', 'Annual_Income', 'Num_of_Delayed_Payment', 'Num_Credit_Card']].copy()

# Clean columns
df.replace('_', np.nan, inplace=True)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(inplace=True)

# Clamp values
df['Age'] = df['Age'].clip(18, 100)
df['Annual_Income'] = df['Annual_Income'].clip(50000, 1e7)
df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].clip(0, 100)
df['Num_Credit_Card'] = df['Num_Credit_Card'].clip(0, 20)

# === Step 2: Generate More Realistic Synthetic Score ===
df['CIBIL_Score'] = (
    900
    - df['Num_of_Delayed_Payment'] * 8                 # Higher penalty
    - df['Num_Credit_Card'] * 2                        # Moderate penalty
    - (df['Age'] < 25).astype(int) * 50               # Penalty for low age
    + (df['Annual_Income'] / 35000)                    # Smaller income boost
    + np.random.normal(0, 20, len(df))                 # Noise
)

# Penalty for very low income
df.loc[df['Annual_Income'] < 200000, 'CIBIL_Score'] -= 50

# Clamp score
df['CIBIL_Score'] = df['CIBIL_Score'].clip(300, 900)

# === Step 3: Split & Train ===
X = df.drop(columns='CIBIL_Score')
y = df['CIBIL_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)

# === Step 4: Evaluate ===
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
within_50 = np.mean(np.abs(y_test - y_pred) <= 50)

print("\n📊 Model Evaluation:")
print(f"✅ MSE   : {mse:.2f}")
print(f"✅ RMSE  : {rmse:.2f}")
print(f"✅ MAE   : {mae:.2f}")
print(f"✅ R²    : {r2:.4f}")
print(f"✅ Accuracy (±50): {within_50 * 100:.2f}%")

# === Step 5: Save Model ===
joblib.dump(model, "cibil_model_final.pkl")
joblib.dump(scaler, "cibil_scaler_final.pkl")

# === Step 6: Sample Prediction ===
print("\n📥 Enter sample user details for prediction:")
age = float(input("Age: "))
income = float(input("Annual Income (₹): "))
delayed = float(input("Number of Delayed Payments: "))
cards = float(input("Number of Credit Cards: "))

# Clamp inputs
age = np.clip(age, 18, 100)
income = np.clip(income, 50000, 1e7)
delayed = np.clip(delayed, 0, 100)
cards = np.clip(cards, 0, 20)

user_df = pd.DataFrame([{
    'Age': age,
    'Annual_Income': income,
    'Num_of_Delayed_Payment': delayed,
    'Num_Credit_Card': cards
}])

user_scaled = scaler.transform(user_df)
score = model.predict(user_scaled)[0]
score = int(np.clip(score, 300, 900))

# Interpretation
if score >= 750:
    category = "Excellent"
    advice = "You are highly eligible for loans and credit cards at low interest."
elif score >= 650:
    category = "Good"
    advice = "You have decent creditworthiness. You might get standard credit offers."
elif score >= 550:
    category = "Fair"
    advice = "You may get limited credit. Improve your score by paying on time."
else:
    category = "Poor"
    advice = "High risk for lenders. Reduce defaults and increase financial discipline."

print(f"\n🎯 Predicted CIBIL Score: {score}")
print(f"📈 Credit Rating: {category}")
print(f"💡 Advice: {advice}")
