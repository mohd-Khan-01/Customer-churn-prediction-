import pandas as pd
import json
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate(model_path, test_dir):

    # ----------------------------
    # 1️⃣ Load model
    # ----------------------------
    model = tf.keras.models.load_model(model_path)

    # ----------------------------
    # 2️⃣ Load test data
    # ----------------------------
    X_test = pd.read_csv(f"{test_dir}/X_test.csv")
    y_test = pd.read_csv(f"{test_dir}/y_test.csv").values.ravel()

    # ----------------------------
    # 3️⃣ Predict probabilities
    # ----------------------------
    probs = model.predict(X_test)

    # Convert probabilities → 0/1
    preds = (probs > 0.5).astype(int).ravel()

    # ----------------------------
    # 4️⃣ Calculate metrics
    # ----------------------------
    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    # ----------------------------
    # 5️⃣ Save metrics
    # ----------------------------
    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation complete ✔")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    evaluate("models/model.keras", "data/processed")