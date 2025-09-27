import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("Loading training data...")
train = pd.read_csv("train_data.txt", sep=":::", engine="python", names=['ID', 'Title', 'Genre', 'Plot'])
train = train.dropna(subset=['Plot', 'Genre'])

X = train['Plot']
y = train['Genre']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = vectorizer.fit_transform(X)

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(X_tfidf, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded)

print("Training Logistic Regression...")
model = LogisticRegression(max_iter=500, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:")
print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

# Save model, vectorizer, and label encoder
joblib.dump(model, "movie_genre_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("Model, vectorizer, and label encoder saved successfully!")
