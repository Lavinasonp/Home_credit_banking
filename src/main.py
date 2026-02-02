import pandas as pd
import numpy as np
import joblib
import gc
from data_preprocessing import preprocess_data
from model_training import train_lgbm_with_cv, find_business_threshold, display_results

def run_pipeline():
    print("--- BANK CREDIT RISK PIPELINE (VERSION 3.0: ENSEMBLE & ALL FEATURES) ---")

    # 1. Define Paths for all 8 required files
    PATHS = {
        'train_path': 'data/application_train.csv',
        'test_path': 'data/application_test.csv',
        'bureau_path': 'data/bureau.csv',
        'bureau_bal_path': 'data/bureau_balance.csv',
        'prev_path': 'data/previous_application.csv',
        'install_path': 'data/installments_payments.csv',
        'pos_path': 'data/POS_CASH_balance.csv',
        'cc_path': 'data/credit_card_balance.csv'
    }

    # 2. Advanced Preprocessing
    # This now includes your original features plus my banking trend enhancements
    X, X_test_final, y = preprocess_data(**PATHS)

    if X is None:
        print("CRITICAL ERROR: Preprocessing failed.")
        return

    print(f"Total features generated: {X.shape[1]}")
    gc.collect()

    # 3. K-Fold Training
    # This fixes the 'Iteration 3' shutdown and provides 5 stable models
    models, oof_preds, importance_df = train_lgbm_with_cv(X, y, n_folds=5)

    # 4. Banking Cost-Benefit Evaluation
    # Finds the threshold that minimizes the financial risk of defaults
    best_threshold = find_business_threshold(y, oof_preds)
    
    # 5. Visualization
    # Displays the confusion matrix and the Gain-based feature importance
    display_results(y, oof_preds, best_threshold, importance_df)

    # 6. Final Inference (Model Averaging / Ensembling)
    print("\nGenerating ensemble predictions for the test set...")
    final_test_preds = np.zeros(X_test_final.shape[0])
    
    for model in models:
        # We sum the probabilities from each fold and divide by the number of folds
        final_test_preds += model.predict_proba(X_test_final)[:, 1] / len(models)
    
    # 7. Create Submission/Result File
    # Load original test IDs to match predictions
    test_ids = pd.read_csv(PATHS['test_path'])['SK_ID_CURR']
    submission = pd.DataFrame({
        'SK_ID_CURR': test_ids,
        'TARGET': final_test_preds
    })
    
    # Save results
    os.makedirs('results', exist_ok=True)
    submission.to_csv('results/submission_ensemble.csv', index=False)
    
    # 8. Save the ensemble model list
    joblib.dump(models, 'models/lgbm_ensemble_v3.pkl')
    
    print("\n--- Pipeline Finished Successfully ---")
    print(f"Final AUC (OOF): {roc_auc_score(y, oof_preds):.4f}")
    print(f"Recommended Decision Threshold: {best_threshold:.2f}")
    print("Ensemble predictions saved to 'results/submission_ensemble.csv'")

if __name__ == "__main__":
    import os
    from sklearn.metrics import roc_auc_score
    run_pipeline()