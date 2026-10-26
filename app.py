
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("models/random_forest.joblib")
FEATURES = list(model.feature_names_in_)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Fraud Detection API is running!",
        "endpoint": "/predict",
        "method": "POST",
        "required_features": len(FEATURES)
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({"error": "Please send a JSON object."}), 400

        missing = [f for f in FEATURES if f not in data]
        if missing:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing
            }), 400

        transaction = pd.DataFrame(
            [[data[f] for f in FEATURES]],
            columns=FEATURES
        )
        transaction = transaction.apply(pd.to_numeric, errors="raise")

        if transaction.isnull().any().any():
            return jsonify({"error": "Feature values cannot be empty."}), 400

        prediction = int(model.predict(transaction)[0])
        probability = float(model.predict_proba(transaction)[0][1])

        return jsonify({
            "prediction": prediction,
            "result": "Potential fraud" if prediction == 1 else "Likely legitimate",
            "fraud_probability": round(probability, 4)
        })

    except (TypeError, ValueError):
        return jsonify({"error": "All feature values must be valid numbers."}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction could not be completed."}), 500


if __name__ == "__main__":
    app.run(debug=True)
