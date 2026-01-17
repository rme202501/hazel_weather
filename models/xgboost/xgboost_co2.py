import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load Data
data = 
X, y = data.data, data.target

# 2. Split Data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize the Model
# use XGBRegressor for predicting continuous numbers (prices, temp, etc.)
model = xgb.XGBClassifier(
    n_estimators=100,      # Number of trees
    learning_rate=0.1,     # How much each tree contributes (step size)
    max_depth=3,           # Depth of each tree (complexity)
    objective='multi:softprob', # Specify the learning task
    random_state=42
)

# 4. Train the Model
model.fit(X_train, y_train)

# 5. Make Predictions
preds = model.predict(X_test)

# 6. Evaluate
accuracy = accuracy_score(y_test, preds)
print(f"Accuracy: {accuracy * 100:.2f}%")