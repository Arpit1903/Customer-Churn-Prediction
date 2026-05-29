
from src.preprocessing import load_and_preprocess
from src.classification import run_classification_models
from src.regression import run_regression_models
from src.visualization import plot_confusion_matrix

def main():

    filepath = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    (
        X_train_c,
        X_test_c,
        y_train_c,
        y_test_c,
        X_train_r,
        X_test_r,
        y_train_r,
        y_test_r,
    ) = load_and_preprocess(filepath)

    print("\n===== CLASSIFICATION MODELS =====\n")

    classification_results = run_classification_models(
        X_train_c,
        X_test_c,
        y_train_c,
        y_test_c
    )

    for model_name, metrics in classification_results.items():

        print(f"\n{model_name}")
        print("-" * 40)

        for metric, value in metrics.items():
            print(f"{metric}: {value}")

        plot_confusion_matrix(
            metrics["Confusion Matrix"],
            model_name.replace(" ", "_")
        )

    print("\n===== REGRESSION MODELS =====\n")

    regression_results = run_regression_models(
        X_train_r,
        X_test_r,
        y_train_r,
        y_test_r
    )

    for model_name, metrics in regression_results.items():

        print(f"\n{model_name}")
        print("-" * 40)

        for metric, value in metrics.items():
            print(f"{metric}: {value}")

if __name__ == "__main__":
    main()
