from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Flask app setup
app = Flask(__name__)
CORS(app)

# Step 1: Load and preprocess the rainfall dataset
try:
    data = pd.read_csv("Rainfall prediction model.csv")  # Ensure file path is correct
except FileNotFoundError:
    raise Exception("Dataset file not found. Please check the file path.")

# Step 2: Handle missing values
data = data.dropna()

# Step 3: Encode categorical variables
categorical_cols = ["res_state", "res_district", "res_name", "rain_month"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Step 4: Define features and target
X = data.drop(columns=["rainfall"])
y = data["rainfall"]

# Step 5: Feature scaling
feature_scaler = MinMaxScaler()
X_scaled = feature_scaler.fit_transform(X)

target_scaler = MinMaxScaler()
y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1))

X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Step 6: Create and train the LSTM model
model = Sequential()
model.add(LSTM(units=64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='linear'))

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Flask Routes
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        print("Received Data:", data)  # Debugging purpose

        state = data.get("state")
        district = data.get("district")
        name = data.get("name")
        year = int(data.get("year"))
        month = data.get("month").lower()  # Normalize month input to lowercase

        # Encode input data with proper error handling
        def safe_encode(value, column):
            if value in label_encoders[column].classes_:
                return label_encoders[column].transform([value])[0]
            else:
                return -1  # Default value for unseen data

        encoded_state = safe_encode(state, "res_state")
        encoded_district = safe_encode(district, "res_district")
        encoded_name = safe_encode(name, "res_name")
        encoded_month = safe_encode(month, "rain_month")

        # Prepare the input for prediction
        input_data = np.array([[encoded_state, encoded_district, encoded_name, year, encoded_month]])
        input_data_scaled = feature_scaler.transform(input_data)
        input_data_scaled = input_data_scaled.reshape((input_data_scaled.shape[0], 1, input_data_scaled.shape[1]))

        # Predict rainfall and inverse transform
        predicted_rainfall_scaled = model.predict(input_data_scaled)
        predicted_rainfall = target_scaler.inverse_transform(predicted_rainfall_scaled)

        print("Predicted Rainfall:", predicted_rainfall)  # Debugging

        return jsonify({"year": year, "predicted_rainfall": float(predicted_rainfall[0][0])})
    except Exception as e:
        print("Error in Prediction:", str(e))  # Debugging
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5002)
