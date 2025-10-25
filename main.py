import os
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import pi

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier


dataset_path = r""  # Choose student-mat.csv or student-por.csv
base_dir = r""
os.makedirs(base_dir, exist_ok=True)

df = pd.read_csv(dataset_path, sep=";")
df["pass"] = (df["G3"] >= 10).astype(int)
y = df["pass"]
X = df.drop(columns=["G1", "G2", "G3", "pass"])
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Split & scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RidgeClassifier": RidgeClassifier(),
    "SGDClassifier": SGDClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "ExtraTrees": ExtraTreesClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "NaiveBayes": GaussianNB(),
    "LDA": LinearDiscriminantAnalysis(),
    "Dummy": DummyClassifier(strategy="most_frequent")
}

summary = []

for name, model in models.items():
    model_dir = os.path.join(base_dir, name)
    os.makedirs(model_dir, exist_ok=True)

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # Some classifiers (like Ridge) don't have predict_proba
        try:
            y_prob = model.decision_function(X_test)
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
        except Exception:
            y_prob = np.zeros_like(y_pred)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob) if len(np.unique(y_prob)) > 1 else np.nan,
        "Train Time (s)": train_time
    }
    summary.append(metrics)

    pd.DataFrame([metrics]).to_csv(os.path.join(model_dir, f"{name}_metrics.csv"), index=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plt.suptitle(f"Model Evaluation - {name}", fontsize=14, weight='bold')
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0, 0])
    axes[0, 0].set_title("Confusion Matrix")
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("True")

    try:
        RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[0, 1])
        axes[0, 1].set_title("ROC Curve")
    except Exception:
        axes[0, 1].text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12)
        axes[0, 1].set_title("ROC Curve")

    try:
        PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=axes[1, 0])
        axes[1, 0].set_title("Precision-Recall Curve")
    except Exception:
        axes[1, 0].text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12)
        axes[1, 0].set_title("Precision-Recall Curve")
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X.columns)
        importances.sort_values(ascending=False).head(10).plot(kind="bar", ax=axes[1, 1], color="steelblue")
        axes[1, 1].set_title("Top 10 Feature Importances")
        axes[1, 1].set_ylabel("Importance")
    else:
        axes[1, 1].text(0.5, 0.5, "Not available", ha="center", va="center", fontsize=12)
        axes[1, 1].set_title("Feature Importances")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(model_dir, f"{name}_summary.png"), dpi=300)
    plt.close()
summary_df = pd.DataFrame(summary).sort_values(by="Accuracy", ascending=False)
summary_df.to_csv(os.path.join(base_dir, "all_models_summary.csv"), index=False)
melted = summary_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
                         var_name="Metric", value_name="Score")
plt.figure(figsize=(12, 6))
sns.barplot(data=melted, x="Model", y="Score", hue="Metric")
plt.title("Performance Comparison of ML Models on Student Performance Dataset", fontsize=14, weight='bold')
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "comparison_metrics.png"), dpi=300)
plt.close()
plt.figure(figsize=(10, 4))
sns.barplot(data=summary_df, x="Model", y="Train Time (s)", color="steelblue")
plt.title("Training Time Comparison", fontsize=14, weight='bold')
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "training_time.png"), dpi=300)
plt.close()
metrics_for_radar = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
values = summary_df[metrics_for_radar].values.tolist()
labels = summary_df["Model"].tolist()
angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False).tolist()
angles += angles[:1]
plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)
for i, vals in enumerate(values):
    vals += vals[:1]
    ax.plot(angles, vals, label=labels[i])
    ax.fill(angles, vals, alpha=0.1)
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics_for_radar)
plt.title("Radar Chart Comparison of ML Models on Student Performance", weight='bold', pad=20)
plt.legend(bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "radar_chart.png"), dpi=300)
plt.close()

print(f"All metrics and graphs saved in: {os.path.abspath(base_dir)}")
print(summary_df.round(4))
