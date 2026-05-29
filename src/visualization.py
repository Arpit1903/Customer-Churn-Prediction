
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, model_name):

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.savefig(f"outputs/{model_name}_confusion_matrix.png")
    plt.close()
