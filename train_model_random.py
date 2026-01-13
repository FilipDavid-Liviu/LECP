import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import config
import os
matplotlib.use('Agg')

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
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # Sort descending

    plt.figure(figsize=(10, 8))
    plt.title("LECP Feature Importance")
    plt.barh(range(len(indices)), importances[indices], align="center", color='green')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.gca().invert_yaxis()  # Highest importance on top

    plt.tight_layout()
    out_path = os.path.join(config.PLOT_DIR, "lecp_feature_importance.png")
    plt.savefig(out_path, dpi=300)
    print(f"Graph saved: {out_path}")
    plt.show()


def plot_four_way_comparison(y_test, preds):
    plt.figure(figsize=(14, 12))
    min_val, max_val = 0, 1

    p1, p2, p3, p4 = preds

    plots = [
        (p1, "Control 1: RBR Only", "red"),
        (p2, "Control 2: PreNBR Only", "orange"),
        (p3, "Control 3: RBR + PreNBR", "purple"),
        (p4, "Control 4: LECP (Spatial Context)", "blue")
    ]

    for i, (pred, title, color) in enumerate(plots):
        plt.subplot(2, 2, i + 1)
        plt.scatter(y_test, pred, alpha=0.1, color=color, s=1)
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        plt.title(title)
        plt.xlabel("Actual Recovery (NDVI)")
        plt.ylabel("Predicted Recovery (NDVI)")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        r2 = r2_score(y_test, pred)
        plt.text(0.05, 0.9, f"R² = {r2:.3f}", transform=plt.gca().transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(config.PLOT_DIR, "four_way_comparison.png")
    plt.savefig(out_path, dpi=300)
    print(f"Comparison Graph saved: {out_path}")


def main():
    data_path = os.path.join(config.TIF_DIR, "training_dataset.csv")
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)

    # 1. Define Target and Groups
    y = df['Target_RecoveryNDVI']
    groups = df['Spatial_Block_ID']

    # 2. Define Feature Sets
    X_c1 = df[['Control_RBR']]
    X_c2 = df[['Control_PreNBR']]
    X_c3 = df[['Control_RBR', 'Control_PreNBR']]
    # LECP uses all patch columns, excluding the target, block ID, and pixel-level controls
    drop_cols = ['Target_RecoveryNDVI', 'Spatial_Block_ID', 'Control_RBR', 'Control_PreNBR']
    X_c4 = df.drop(columns=drop_cols)

    # 3. SPATIAL SPLIT (GroupShuffleSplit)
    # This replaces train_test_split to prevent data leakage
    print(f"Performing Spatial Split (GroupShuffleSplit) on {len(df['Spatial_Block_ID'].unique())} blocks...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(X_c4, y, groups=groups))

    # Apply indices to all feature sets and target
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    X_train_c1, X_test_c1 = X_c1.iloc[train_idx], X_c1.iloc[test_idx]
    X_train_c2, X_test_c2 = X_c2.iloc[train_idx], X_c2.iloc[test_idx]
    X_train_c3, X_test_c3 = X_c3.iloc[train_idx], X_c3.iloc[test_idx]
    X_train_c4, X_test_c4 = X_c4.iloc[train_idx], X_c4.iloc[test_idx]

    print(f"Train size: {len(y_train)} | Test size: {len(y_test)}")

    # 4. Train & Evaluate
    models = [
        (X_train_c1, X_test_c1, "Control 1 (RBR Only)"),
        (X_train_c2, X_test_c2, "Control 2 (PreNBR Only)"),
        (X_train_c3, X_test_c3, "Control 3 (Pixel Combo)"),
        (X_train_c4, X_test_c4, "Control 4 (LECP)")
    ]

    all_preds = []
    final_lecp_model = None

    for X_tr, X_te, name in models:
        print(f"\nTraining {name}...")
        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, min_samples_leaf=4, random_state=SEED)
        rf.fit(X_tr, y_train)
        preds = rf.predict(X_te)
        evaluate_model(y_test, preds, name)
        all_preds.append(preds)

        if "LECP" in name:
            final_lecp_model = rf

    plot_four_way_comparison(y_test, all_preds)
    plot_feature_importance(final_lecp_model, X_c4.columns)


if __name__ == "__main__":
    main()
