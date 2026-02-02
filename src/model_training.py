import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

def train_lgbm_with_cv(X, y, n_folds=5):
    """Trains LightGBM using Stratified K-Fold and stabilized parameters."""
    print(f"\n--- Starting {n_folds}-Fold Cross-Validation ---")
    
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(X.shape[0])
    feature_importance_df = pd.DataFrame()
    models = []

    # Calculate scale_pos_weight to handle class imbalance
    # Ratio of Negatives to Positives (usually ~11.0 in this dataset)
    ratio = y.value_counts()[0] / y.value_counts()[1]

    for n_fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # STABILIZED PARAMETERS to prevent early stopping at iteration 3
        model = LGBMClassifier(
            n_estimators=20000,        # Massive room to learn
            learning_rate=0.005,       # Tiny steps to prevent "overshooting"
            num_leaves=70,             # Complex trees
            max_depth=10,              # Deeper trees
            subsample=0.85,            
            colsample_bytree=0.8,      
            min_child_samples=20,      
            reg_alpha=0.01,            
            reg_lambda=0.01,           
            scale_pos_weight=ratio,    
            random_state=42,
            importance_type='gain',
            verbose=-1
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            # INCREASED PATIENCE: Won't stop unless no improvement for 500 rounds
            callbacks=[
                early_stopping(stopping_rounds=500, verbose=True), 
                log_evaluation(period=500)
            ]
        )

        # Store predictions for the validation fold
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        
        # Track feature importance across folds
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = X.columns
        fold_importance_df["importance"] = model.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df])
        
        models.append(model)
        print(f"Fold {n_fold + 1} AUC : {roc_auc_score(y_val, oof_preds[val_idx]):.6f}")

    print(f"\nFull Out-of-Fold (OOF) AUC score: {roc_auc_score(y, oof_preds):.6f}")
    return models, oof_preds, feature_importance_df

def find_business_threshold(y_true, y_probs):
    """
    Finds a threshold by minimizing financial cost.
    Assumes missing a default (FN) is 10x more costly than rejecting a good client (FP).
    """
    thresholds = np.linspace(0, 1, 101)
    costs = []
    
    cost_fn = 10  # Cost of a bad loan
    cost_fp = 1   # Cost of a lost opportunity
    
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        costs.append(total_cost)
        
    best_threshold = thresholds[np.argmin(costs)]
    print(f"Optimal Banking Threshold: {best_threshold:.2f}")
    return best_threshold

def display_results(y_true, y_probs, threshold, importance_df):
    """Visualizes model performance and the top driving features."""
    y_pred = (y_probs >= threshold).astype(int)
    
    plt.figure(figsize=(15, 6))
    
    # Confusion Matrix
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix (Threshold {threshold:.2f})")
    
    # Feature Importance (Gain)
    plt.subplot(1, 2, 2)
    best_features = importance_df.groupby("feature")["importance"].mean().sort_values(ascending=False).head(20)
    sns.barplot(x=best_features.values, y=best_features.index, palette='viridis')
    plt.title("Top 20 Features by Gain (Mean across Folds)")
    
    plt.tight_layout()
    plt.show()
    
    print("\nFinal Classification Report:")
    print(classification_report(y_true, y_pred))