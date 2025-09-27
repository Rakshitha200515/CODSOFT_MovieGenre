from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# Load model, vectorizer, label encoder
model = joblib.load("movie_genre_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    plot = data.get("plot", "")
    if not plot:
        return jsonify({"error": "No plot provided"}), 400

    X_test = vectorizer.transform([plot])
    pred_encoded = model.predict(X_test)[0]
    pred_prob = model.predict_proba(X_test)[0]

    # Get top 5 predictions
    top_indices = pred_prob.argsort()[::-1][:5]
    top = [{"genre": label_encoder.classes_[i], "prob": pred_prob[i]} for i in top_indices]

    return jsonify({"genre": label_encoder.inverse_transform([pred_encoded])[0], "top": top})

if __name__ == "__main__":
    print("Movie Genre Classification API is running!")
    app.run(debug=True)
