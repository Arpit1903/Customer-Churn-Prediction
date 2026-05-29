
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    results = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(y_test, predictions),
        "Confusion Matrix": confusion_matrix(y_test, predictions),
    }

    return results

def run_classification_models(X_train, X_test, y_train, y_test):

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
    }

    model_results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        model_results[name] = evaluate_model(model, X_test, y_test)

    return model_results
