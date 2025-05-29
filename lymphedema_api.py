from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import traceback

from fix_model import preprocess_input

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "lymphedema_model.pkl")
model = joblib.load(model_path)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from the request
        data = request.json

        # Convert to dataframe
        df = pd.DataFrame([data])

        # Preprocess the data using the function from fix_model.py
        processed_data = preprocess_input(df)

        # Make prediction
        probability = model.predict_proba(processed_data)[0][1]
        prediction = int(probability >= 0.5)

        # Determine risk category
        risk_category = "Low"
        if probability >= 0.75:
            risk_category = "Very High"
        elif probability >= 0.6:
            risk_category = "High"
        elif probability >= 0.4:
            risk_category = "Moderate"

        return jsonify(
            {
                "success": True,
                "probability": float(probability),
                "prediction": prediction,
                "risk_category": risk_category,
            }
        )

    except Exception as e:
        print(f"Error during data processing or prediction: {str(e)}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    print("Starting Lymphedema API server...")
    app.run(debug=True)
