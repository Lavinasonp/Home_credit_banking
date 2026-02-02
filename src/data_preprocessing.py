import pandas as pd
import numpy as np
import re
import gc

def handle_anomalies_and_feature_eng(df):
    """Unified anomalies fix and expanded domain features."""
    # Fix the "1000 years" anomaly
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    
    # Convert DAYS to YEARS (kept for clarity as per your original)
    df['YEARS_BIRTH'] = abs(df['DAYS_BIRTH']) / 365
    df['YEARS_EMPLOYED'] = abs(df['DAYS_EMPLOYED']) / 365
    
    # Financial Ratios (Fixed denominators and logic)
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1e-6)
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1e-6)
    df['ANNUITY_CREDIT_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1e-6)
    
    # New: Disposable Income
    df['INCOME_MINUS_ANNUITY'] = df['AMT_INCOME_TOTAL'] - df['AMT_ANNUITY']
    
    # Drop original days to keep the feature space clean as per your original logic
    df.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1, inplace=True)
    return df

def process_bureau_data(bureau_path, bureau_bal_path):
    """Processes bureau.csv with your original aggs + bureau_balance trends."""
    print("Processing bureau and bureau balance...")
    # Bureau Balance trends
    bb = pd.read_csv(bureau_bal_path)
    bb = pd.get_dummies(bb, columns=['STATUS'], prefix='BB_STATUS')
    bb_agg = bb.groupby('SK_ID_BUREAU').agg({'MONTHS_BALANCE': ['min', 'max', 'size']})
    bb_agg.columns = pd.Index(['BB_' + e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    
    bureau = pd.read_csv(bureau_path)
    bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')
    
    # Fix anomalies
    for col in ['DAYS_CREDIT', 'DAYS_CREDIT_UPDATE']:
        bureau[col] = bureau[col].replace(365243, np.nan)

    # Your original aggregations + additional bureau insights
    agg_dict = {
        'DAYS_CREDIT': ['mean', 'max', 'min'],
        'CREDIT_DAY_OVERDUE': ['mean', 'max'],
        'AMT_CREDIT_SUM': ['mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
        'AMT_ANNUITY': ['mean', 'sum'],
        'CNT_CREDIT_PROLONG': ['sum'],
    }
    
    # Capture the categorical distribution (Credit Type, etc.)
    cat_features = [col for col in bureau.columns if bureau[col].dtype == 'object']
    bureau = pd.get_dummies(bureau, columns=cat_features, dummy_na=True)
    
    for col in [c for c in bureau.columns if 'NAME' in c or 'TYPE' in c or 'STATUS' in c]:
        agg_dict[col] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg(agg_dict)
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    
    del bureau, bb; gc.collect()
    return bureau_agg

def process_previous_applications(prev_path):
    """Your original features + Last Status Refused logic."""
    print("Processing previous applications...")
    prev = pd.read_csv(prev_path)
    for col in ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']:
        prev[col] = prev[col].replace(365243, np.nan)

    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / (prev['AMT_CREDIT'] + 1e-6)
    
    # Keep your original aggs
    agg_dict = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    
    prev_cat = [col for col in prev.columns if prev[col].dtype == 'object']
    prev = pd.get_dummies(prev, columns=prev_cat, dummy_na=True)
    
    for col in [c for c in prev.columns if 'NAME' in c or 'CODE' in c]:
        agg_dict[col] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg(agg_dict)
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    
    # Add the "Last Loan Status" Refusal indicator
    last_status = prev.sort_values(by=['SK_ID_CURR', 'DAYS_DECISION']).groupby('SK_ID_CURR').last()
    if 'NAME_CONTRACT_STATUS_Refused' in last_status.columns:
        prev_agg['PREV_LAST_STATUS_REFUSED'] = last_status['NAME_CONTRACT_STATUS_Refused'].fillna(0)
    
    del prev; gc.collect()
    return prev_agg

def process_installments_payments(install_path):
    """Your original engineering: DIFF, Past Due, and Under Payment."""
    print("Processing installments...")
    install = pd.read_csv(install_path)
    
    install['PAYMENT_PERC'] = install['AMT_PAYMENT'] / (install['AMT_INSTALMENT'] + 1e-6)
    install['PAYMENT_DIFF'] = install['AMT_INSTALMENT'] - install['AMT_PAYMENT']
    install['DAYS_PAST_DUE'] = (install['DAYS_ENTRY_PAYMENT'] - install['DAYS_INSTALMENT']).clip(lower=0)
    install['DAYS_BEFORE_DUE'] = (install['DAYS_INSTALMENT'] - install['DAYS_ENTRY_PAYMENT']).clip(lower=0)
    install['LATE_PAYMENT'] = (install['DAYS_PAST_DUE'] > 0).astype(int)
    install['UNDER_PAYMENT'] = (install['PAYMENT_DIFF'] > 0).astype(int)
    
    agg_dict = {
        'PAYMENT_PERC': ['mean', 'max', 'var'],
        'PAYMENT_DIFF': ['mean', 'max', 'sum'],
        'DAYS_PAST_DUE': ['mean', 'max', 'sum'],
        'DAYS_BEFORE_DUE': ['mean', 'min'],
        'LATE_PAYMENT': ['mean', 'sum'],
        'UNDER_PAYMENT': ['mean', 'sum'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum']
    }
    
    install_agg = install.groupby('SK_ID_CURR').agg(agg_dict)
    install_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in install_agg.columns.tolist()])
    
    del install; gc.collect()
    return install_agg

def process_pos_cash_data(pos_path):
    """Your original aggs + delinquency counts."""
    print("Processing POS_CASH...")
    pos = pd.read_csv(pos_path)
    pos = pd.get_dummies(pos, columns=['NAME_CONTRACT_STATUS'], dummy_na=True)
    
    agg_dict = {
        'MONTHS_BALANCE': ['mean', 'max', 'min', 'size'],
        'SK_DPD': ['mean', 'max', 'sum'],
        'SK_DPD_DEF': ['mean', 'max', 'sum']
    }
    
    cat_cols = [col for col in pos.columns if 'NAME_CONTRACT_STATUS' in col]
    for col in cat_cols:
        agg_dict[col] = ['mean']
        
    pos_agg = pos.groupby('SK_ID_CURR').agg(agg_dict)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    
    del pos; gc.collect()
    return pos_agg

def process_credit_card_data(cc_path):
    """Your original credit utilization logic."""
    print("Processing Credit Card Balance...")
    cc = pd.read_csv(cc_path)
    cc = pd.get_dummies(cc, columns=['NAME_CONTRACT_STATUS'], dummy_na=True)
    
    cc['CREDIT_UTILIZATION'] = cc['AMT_BALANCE'] / (cc['AMT_CREDIT_LIMIT_ACTUAL'] + 1e-6)
    
    agg_dict = {
        'MONTHS_BALANCE': ['mean', 'max', 'min', 'size'],
        'AMT_BALANCE': ['mean', 'max', 'min', 'sum'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['mean', 'max'],
        'AMT_DRAWINGS_CURRENT': ['mean', 'max', 'sum'],
        'AMT_DRAWINGS_ATM_CURRENT': ['mean', 'max', 'sum'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['mean', 'max', 'sum'],
        'CNT_DRAWINGS_CURRENT': ['mean', 'max', 'sum'],
        'SK_DPD': ['mean', 'max', 'sum'],
        'CREDIT_UTILIZATION': ['mean', 'max', 'min']
    }
    
    cc_agg = cc.groupby('SK_ID_CURR').agg(agg_dict)
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    
    del cc; gc.collect()
    return cc_agg

def preprocess_data(train_path, test_path, bureau_path, bureau_bal_path, prev_path, install_path, pos_path, cc_path):
    """Full pipeline: Main data + All 6 external tables."""
    print("Starting Comprehensive Preprocessing...")
    df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    df = pd.concat([df, test_df], axis=0, ignore_index=True)
    
    df = handle_anomalies_and_feature_eng(df)
    
    # Sequential Merges
    df = df.merge(process_bureau_data(bureau_path, bureau_bal_path), on='SK_ID_CURR', how='left')
    df = df.merge(process_previous_applications(prev_path), on='SK_ID_CURR', how='left')
    df = df.merge(process_pos_cash_data(pos_path), on='SK_ID_CURR', how='left')
    df = df.merge(process_installments_payments(install_path), on='SK_ID_CURR', how='left')
    df = df.merge(process_credit_card_data(cc_path), on='SK_ID_CURR', how='left')
    
    # Encoding & Alignment
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_cols, dummy_na=True)
    
    train_df = df[df['TARGET'].notnull()].copy()
    test_df = df[df['TARGET'].isnull()].copy()
    train_labels = train_df['TARGET']
    
    train_df.drop(['TARGET', 'SK_ID_CURR'], axis=1, inplace=True)
    test_df.drop(['TARGET', 'SK_ID_CURR'], axis=1, inplace=True)
    
    # Missing value indicators for the big three
    for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
        train_df[f'{col}_NAN'] = train_df[col].isnull().astype(int)
        test_df[f'{col}_NAN'] = test_df[col].isnull().astype(int)

    # Impute medians (Train only)
    medians = train_df.median()
    train_df.fillna(medians, inplace=True)
    test_df.fillna(medians, inplace=True)
    
    # Sanitize names for LightGBM
    train_df.columns = [re.sub(r'[^A-Za-z0-9_]+', '', c) for c in train_df.columns]
    test_df.columns = [re.sub(r'[^A-Za-z0-9_]+', '', c) for c in test_df.columns]
    
    return train_df, test_df, train_labels