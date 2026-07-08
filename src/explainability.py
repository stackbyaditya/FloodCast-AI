import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def generate_shap_plots(
    pipeline,
    X_train,
    output_dir="../outputs/figures"
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_processed = preprocessor.transform(X_train)

    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_processed)

    # ---------------- Summary Plot ---------------- #

    plt.figure(figsize=(12,8))

    shap.summary_plot(
        shap_values,
        X_processed,
        feature_names=feature_names,
        show=False
    )

    plt.savefig(
        output_dir / "shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ---------------- Bar Plot ---------------- #

    plt.figure(figsize=(12,8))

    shap.summary_plot(
        shap_values,
        X_processed,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )

    plt.savefig(
        output_dir / "shap_bar.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ---------------- Waterfall ---------------- #

    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_processed[0],
        feature_names=feature_names
    )

    shap.plots.waterfall(
        explanation,
        show=False
    )

    plt.savefig(
        output_dir / "waterfall.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("✅ SHAP plots saved.")


def save_feature_importance(
    pipeline,
    X_train,
    output_path="../outputs/reports/feature_importance.csv"
):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    importance.to_csv(
        output_path,
        index=False
    )

    print(f"✅ Feature importance saved to {output_path}")

    return importance