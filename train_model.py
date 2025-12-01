import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import config
import os
matplotlib.use('Agg')

# --- SETTINGS ---
# Random seed for reproducibility
SEED = 42


def evaluate_model(y_true, y_pred, model_name):
    """Calculates standard regression metrics."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print(f"--- {model_name} Performance ---")
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE:     {rmse:.4f}")
    print(f"MAE:      {mae:.4f}")
    print("-" * 30)
    return r2, rmse, mae


def plot_feature_importance(model, feature_names):
    """
    Bar chart showing which features the Random Forest found most useful.
    This validates the hypothesis that context matters.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # Sort descending

    plt.figure(figsize=(10, 8))
    plt.title("LECP Feature Importance")
    plt.barh(range(len(indices)), importances[indices], align="center", color='green')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.gca().invert_yaxis()  # Highest importance on top

    plt.tight_layout()
    out_path = os.path.join(config.BASE_DIR, "lecp_feature_importance.png")
    plt.savefig(out_path, dpi=300)
    print(f"Graph saved: {out_path}")
    plt.show()


def plot_four_way_comparison(y_test, preds):
    plt.figure(figsize=(14, 12))
    min_val, max_val = 0, 1

    p1, p2, p3, p4 = preds

    plots = [
        (p1, "Control 1: RBR Only\n(Severity / Trauma)", "red"),
        (p2, "Control 2: PreNBR Only\n(Initial Health / Memory)", "orange"),
        (p3, "Control 3: RBR + PreNBR\n(Pixel-Level Combo)", "purple"),
        (p4, "Control 4: LECP\n(Combo + Spatial Neighbors)", "blue")
    ]

    for i, (pred, title, color) in enumerate(plots):
        plt.subplot(2, 2, i + 1)
        plt.scatter(y_test, pred, alpha=0.1, color=color, s=1)
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        plt.title(title)
        plt.xlabel("Actual Recovery")
        plt.ylabel("Predicted Recovery")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)

        r2 = r2_score(y_test, pred)
        plt.text(0.05, 0.9, f"R² = {r2:.3f}", transform=plt.gca().transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(config.BASE_DIR, "four_way_comparison.png")
    plt.savefig(out_path, dpi=300)
    print(f"Comparison Graph saved: {out_path}")


def main():
    data_path = os.path.join(config.BASE_DIR, "training_dataset.csv")
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)

    y = df['Target_RecoveryNDVI']

    # --- DEFINE FEATURE SETS ---

    # Control 1: RBR Only (Severity)
    X_c1 = df[['Control_RBR']]

    # Control 2: PreNBR Only (Memory)
    X_c2 = df[['Control_PreNBR']]

    # Control 3: RBR + PreNBR (Pixel Combo)
    X_c3 = df[['Control_RBR', 'Control_PreNBR']]

    # Control 4: LECP (Hypothesis)
    # We drop the explicit control columns to force it to use the 18 patch columns
    # Note: The patch columns INCLUDE the center pixel info, so it has access to C3 info + neighbors
    X_c4 = df.drop(columns=['Target_RecoveryNDVI', 'Control_RBR', 'Control_PreNBR'])

    # --- SPLIT ---
    print("Splitting data...")
    X_train_c4, X_test_c4, y_train, y_test = train_test_split(
        X_c4, y, test_size=0.2, random_state=SEED
    )

    # Sync others
    X_train_c1, X_test_c1 = X_c1.loc[X_train_c4.index], X_c1.loc[X_test_c4.index]
    X_train_c2, X_test_c2 = X_c2.loc[X_train_c4.index], X_c2.loc[X_test_c4.index]
    X_train_c3, X_test_c3 = X_c3.loc[X_train_c4.index], X_c3.loc[X_test_c4.index]

    # --- TRAIN & EVALUATE ---

    print("\nTraining Control 1 (RBR Only)...")
    rf1 = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=SEED)
    rf1.fit(X_train_c1, y_train)
    pred1 = rf1.predict(X_test_c1)
    evaluate_model(y_test, pred1, "Control 1 (RBR Only)")

    print("\nTraining Control 2 (PreNBR Only)...")
    rf2 = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=SEED)
    rf2.fit(X_train_c2, y_train)
    pred2 = rf2.predict(X_test_c2)
    evaluate_model(y_test, pred2, "Control 2 (PreNBR Only)")

    print("\nTraining Control 3 (Pixel Combo)...")
    rf3 = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=SEED)
    rf3.fit(X_train_c3, y_train)
    pred3 = rf3.predict(X_test_c3)
    evaluate_model(y_test, pred3, "Control 3 (RBR + PreNBR)")

    print("\nTraining Control 4 (LECP)...")
    rf4 = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=SEED)
    rf4.fit(X_train_c4, y_train)
    pred4 = rf4.predict(X_test_c4)
    evaluate_model(y_test, pred4, "Control 4 (LECP)")

    # --- PLOTTING ---
    plot_four_way_comparison(y_test, (pred1, pred2, pred3, pred4))
    plot_feature_importance(rf4, X_c4.columns)


if __name__ == "__main__":
    main()
