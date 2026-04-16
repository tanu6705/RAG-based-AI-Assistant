import pickle
import joblib

# Load existing pickle file
with open("embeddings_db.pkl", "rb") as f:
    data = pickle.load(f)

# Save as joblib
joblib.dump(data, "embeddings.joblib")

print("Converted successfully!")