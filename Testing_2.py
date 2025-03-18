'''
WORKING WITH THE IMPROVEMENT CODE
'''

'''
IMPORTING THE NECESSARY LIBRARIES
'''
#LIBRARIES
import pandas as pd
import numpy as np 
import ta 
import xgboost as xgb 
import lightgbm as lgbm
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm as tqdm
import warnings
import gc
import os

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier


print('SCRIPT HAS STARTED')
'''
LOAD THE DATA
'''
df = pd.read_csv('BTC_1min.csv')

'''
GARBAGE COLLECTION
'''
gc.enable()

#XGBoost 
os.environ['OMP_NUM_THEARDS'] = '2'
warnings.filterwarnings('ignore')


# 1. Load and Prepare Data with improved memory management
def load_and_prepare_data(file_path, sample_rate=0.25, chunksize=100000):
    print("Loading data...")
    # Read with optimized dtypes to reduce memory usage
    dtypes = {
        'Open': 'float32',
        'High': 'float32', 
        'Low': 'float32',
        'Close': 'float32',
        'Volume': 'float32'
    }
    
    # Use chunksize to read data in chunks
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunksize, dtype=dtypes):
        # Sample only a portion of the data
        if sample_rate < 1.0:
            chunk = chunk.sample(frac=sample_rate)
        chunks.append(chunk)
        
    df = pd.concat(chunks)
    
    # Convert timestamp to datetime
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
    
    # Add basic price features
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1)).astype('float32')
    df['hl_ratio'] = (df['High'] / df['Low']).astype('float32')
    df['co_ratio'] = (df['Close'] / df['Open']).astype('float32')
    df['hc_ratio'] = (df['High'] / df['Close']).astype('float32')
    df['lc_ratio'] = (df['Low'] / df['Close']).astype('float32')
    df['price_range'] = ((df['High'] - df['Low']) / df['Close']).astype('float32')
    
    # Add time-based features
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    
    # Handle isocalendar week properly to avoid NaNs
    df['week_of_year'] = df.index.isocalendar().week
    df['week_of_year'] = df['week_of_year'].fillna(0).astype('int16')
    
    df['month'] = df.index.month
    
    # Identify gaps in the 1-minute data
    df['time_diff'] = df.index.to_series().diff().dt.total_seconds() / 60
    df['gap'] = df['time_diff'].gt(1).fillna(False).astype('int8')
    
    # Convert all float64 to float32
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    # Replace infinities with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Forward fill NaN values
    df = df.fillna(method='ffill')
    df = df.fillna(0)
    
    print("Data preparation complete.")
    print(f"DataFrame memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    return df
    
'''
FEATURE ENGINEERING
'''
from joblib import Parallel, delayed

def compute_technical_indicator(df, window, col_name, func):
    return func(df['Close'], window=window).astype('float32')
    

def engineer_features(df, window_sizes=[5, 15, 30]):
    print("Engineering features...")
    df_feat = df.copy()
    
    # Parallelize feature computation
    tasks = []
    for window in window_sizes:
        tasks.extend([
            (df_feat, window, f'sma_{window}', ta.trend.sma_indicator),
            (df_feat, window, f'ema_{window}', ta.trend.ema_indicator),
            (df_feat, window, f'rsi_{window}', ta.momentum.rsi),
            # Add other indicators as needed
        ])
    
    results = Parallel(n_jobs=-1)(delayed(compute_technical_indicator)(*task) for task in tasks)
    for (window, col_name), result in zip([(t[1], t[2]) for t in tasks], results):
        df_feat[col_name] = result
    
    # Non-parallel features
    df_feat['macd'] = ta.trend.macd(df_feat['Close']).astype('float32')
    
    '''
    ADDING MORE FEATURES
    '''
    df_feat['macd_signal'] = ta.trend.macd_signal(df_feat['Close']).astype('float32')
    df_feat['macd_diff'] = ta.trend.macd_diff(df_feat['Close']).astype('float32')
    df_feat['volume_mean_ratio'] = (df_feat['Volume'] / df_feat['Volume'].rolling(window=30).mean()).astype('float32')
    
    window = 30
    df_feat[f'support_{window}'] = df_feat['Low'].rolling(window=window).min().astype('float32')
    df_feat[f'resistance_{window}'] = df_feat['High'].rolling(window=window).max().astype('float32')
    df_feat[f'price_to_support_{window}'] = ((df_feat['Close'] - df_feat[f'support_{window}']) / df_feat[f'support_{window}']).astype('float32')
    df_feat[f'price_to_resistance_{window}'] = ((df_feat['Close'] - df_feat[f'resistance_{window}']) / df_feat[f'resistance_{window}']).astype('float32')
    
    
    
    df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_feat = df_feat.fillna(method='ffill').fillna(0)
    gc.collect()
    
    print(f"Feature engineering complete. Created {len(df_feat.columns) - len(df.columns)} new features.")
    return df_feat
print('FEATURE DONE')


# 3. Target Variable Creation
def create_targets(df, horizons=[5, 15]):
    print("Creating target variables...")
    
    for horizon in horizons:
        future_return = df['Close'].pct_change(horizon).shift(-horizon)
        df[f'target_{horizon}'] = future_return.gt(0).fillna(False).astype('int8')
        
        if horizon == horizons[0]:
            bins = [-np.inf, -0.005, -0.0001, 0.0001, 0.005, np.inf]
            future_return_filled = future_return.fillna(0)
            labels = pd.cut(future_return_filled, bins=bins, labels=[0, 1, 2, 3, 4])
            numeric_labels = pd.to_numeric(labels, errors='coerce')
            df[f'target_class_{horizon}'] = numeric_labels.fillna(0).astype('int8')
    
    print("Target creation complete.")
    return df

'''
FEATURE SELECTION
'''
import pickle

def select_features(df, target_col, correlation_threshold=0.95, min_importance=0.005, max_features=40):
    print("Selecting features...")
    
    quasi_constant_feat = []
    for feat in df.columns:
        if df[feat].nunique() <= 2:
            unique_counts = df[feat].value_counts(normalize=True)
            if unique_counts.max() > 0.99:
                quasi_constant_feat.append(feat)
    
    print(f"Removing {len(quasi_constant_feat)} quasi-constant features")
    df_filtered = df.drop(columns=quasi_constant_feat)
    
    features = [col for col in df_filtered.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume'] and not col.startswith('target_')]
    
    if len(df_filtered) > 100000:
        corr_sample = df_filtered.sample(n=100000, random_state=42)
    else:
        corr_sample = df_filtered
    
    corr_with_target = corr_sample[features + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
    low_corr_features = corr_with_target[corr_with_target < 0.01].index.tolist()
    
    corr_matrix = corr_sample[features].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
    
    print(f"Removing {len(to_drop)} highly correlated features")
    final_features = [f for f in features if f not in to_drop + low_corr_features]
    
    if len(final_features) > max_features:
        if len(df_filtered) > 50000:
            importance_sample = df_filtered.sample(n=50000, random_state=42)
        else:
            importance_sample = df_filtered
            
        X = importance_sample[final_features].values
        y = importance_sample[target_col].values
        
        model = lgbm.LGBMClassifier(n_estimators=100, random_state=42, use_missing=False, zero_as_missing=False)
        model.fit(X, y)
        
        importances = pd.DataFrame({'feature': final_features, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
        important_features = importances[importances['importance'] > min_importance]['feature'].tolist()
        if len(important_features) > max_features:
            important_features = important_features[:max_features]
            
        print(f"Selecting {len(important_features)} features based on importance")
        final_features = important_features
    
    gc.collect()
    
    # Debugging print statements
    print(f"Selected features: {final_features}")
    print(f"Number of selected features: {len(final_features)}")
    
    print(f"Feature selection complete. Selected {len(final_features)} features.")
    return df_filtered, final_features
print('SELECTION DONE')
    
'''
FEATURES TRAINING 
'''
def train_and_evaluate_models(df, features, target_col, horizon, cv_splits=3, max_train_size=100000):
    print(f"Training models for {target_col} (horizon: {horizon})...")
    
    if len(df) > max_train_size:
        print(f"Sampling {max_train_size} rows from {len(df)} total rows")
        train_df = df.sample(n=max_train_size, random_state=42)
    else:
        train_df = df
    
    # Validate features exist in train_df
    missing = [f for f in features if f not in train_df.columns]
    if missing:
        raise ValueError(f"Features missing in train_df: {missing}")
    
    X = train_df[features].values
    y = train_df[target_col].values
    
    print(f"Training features: {features}")
    print(f"Number of features in X: {X.shape[1]}")
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Store the features used for scaling
    scaler.feature_names_ = features  # Custom attribute for validation
    print(f"Scaler fitted on {len(scaler.feature_names_)} features: {scaler.feature_names_}")
    
    # Model training (unchanged for brevity)
    models = {
        'LightGBM': lgbm.LGBMClassifier( n_estimators=200, learning_rate=0.01, max_depth=6, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, class_weight='balanced',
            random_state=42, use_missing=False, zero_as_missing=False, force_col_wise=True),
        'XGBoost': xgb.XGBClassifier(n_estimators=200, learning_rate=0.01, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=sum(y == 0) / sum(y == 1),
            random_state=42, tree_method='hist')
    }
    
    tscv=TimeSeriesSplit(n_splits=cv_splits)
    
    for name, model in models.items():
        print(f"Training {name}...")
        fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            fold_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            fold_metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            fold_metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            fold_metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
            fold_metrics['auc'].append(roc_auc_score(y_test, y_proba))
            
            gc.collect()
        results = { 
                'accuracy': np.mean(fold_metrics['accuracy']),
                'precision': np.mean(fold_metrics['precision']),
                'recall': np.mean(fold_metrics['recall']),
                'f1': np.mean(fold_metrics['f1']),
                'auc': np.mean(fold_metrics['auc']),
                'model': model
            }
    print(type(results))
    if isinstance(results, dict):
        print("Keys in results:", list(results.keys()))
        for key in results:
            print(f"Type of results[{key}] is {type(results[key])}")        
    best_metrics = {k: v for k, v in results.items() if k != 'model'}
    best_model_name_name = results['model']
    
    print(f"Best model: {best_model_name_name} - Accuracy: {best_metrics['accuracy']:.4f}, AUC: {best_metrics['auc']:.4f}")
    return  best_model_name_name, best_metrics, scaler
print('TRAINING DONE')

'''
BACKTEST
''' 
def backtest_strategy(df, model, features, target_col, scaler, horizon=5, commission_rate=0.001, chunk_size=100000):
    print(f"Backtesting trading strategy...")
    
    #TRIYING TO FIX THE MISSING LOG 
    if 'log_return' not in df.columns and 'Close' in df.columns:
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        print("Added 'log_return' to df")
    
    # Avoid duplicates in needed_columns
    base_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'log_return']
    needed_columns = list(dict.fromkeys(base_cols + features))  # Ensures uniqueness
    backtest_df = df[needed_columns].copy()
    
    # Validate features against scaler
    if hasattr(scaler, 'feature_names_'):
        if set(features) != set(scaler.feature_names_) or len(features) != len(scaler.feature_names_):
            raise ValueError(
                f"Feature mismatch: backtesting uses {len(features)} features {features}, "
                f"but scaler expects {len(scaler.feature_names_)} features {scaler.feature_names_}"
            )
    
    if len(backtest_df) > chunk_size:
        chunks = []
        for i in range(0, len(backtest_df), chunk_size):
            chunk = backtest_df.iloc[i:i+chunk_size].copy()
            X_chunk = chunk[features].values
            
            print(f"Backtesting features: {features}")
            print(f"Number of features in X_chunk: {X_chunk.shape[1]}")
            
            # Additional validation
            if X_chunk.shape[1] != scaler.n_features_in_:
                raise ValueError(
                    f"X_chunk has {X_chunk.shape[1]} features, but scaler expects {scaler.n_features_in_}"
                )
            
            X_scaled_chunk = scaler.transform(X_chunk)
            chunk['model_prob'] = model.predict_proba(X_scaled_chunk)[:, 1]
            chunks.append(chunk)
            gc.collect()
        backtest_df = pd.concat(chunks)
    else:
        X = backtest_df[features].values
        print(f"Backtesting features: {features}")
        print(f"Number of features in X: {X.shape[1]}")
        
        if X.shape[1] != scaler.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but scaler expects {scaler.n_features_in_}"
            )
        
        X_scaled = scaler.transform(X)
        backtest_df['model_prob'] = model.predict_proba(X_scaled)[:, 1]
    
    # Backtesting logic remains the same
    # ... (omitted for brevity)
    
    
    '''
    TRYING ADDING MORE COMPLEXITY
    '''
    backtest_df['position'] = 0
    backtest_df.loc[backtest_df['model_prob'] > 0.6, 'position'] = 1
    backtest_df.loc[backtest_df['model_prob'] < 0.4, 'position'] = -1
    backtest_df['position'] = backtest_df['position'].shift(1)
    backtest_df['position'] = backtest_df['position'].fillna(0)
    
    backtest_df['strategy_return'] = backtest_df['position'] * backtest_df['log_return']
    backtest_df['position_change'] = backtest_df['position'].diff().abs()
    backtest_df['position_change'] = backtest_df['position_change'].fillna(0)
    backtest_df['transaction_cost'] = backtest_df['position_change'] * commission_rate
    backtest_df['strategy_return_net'] = backtest_df['strategy_return'] - backtest_df['transaction_cost']
    
    backtest_df['cum_market_return'] = np.exp(backtest_df['log_return'].cumsum()) - 1
    backtest_df['cum_strategy_return'] = np.exp(backtest_df['strategy_return_net'].cumsum()) - 1
    
    total_trades = backtest_df['position_change'].sum()
    winning_trades = backtest_df[backtest_df['strategy_return_net'] > 0]['position_change'].sum()
    losing_trades = backtest_df[backtest_df['strategy_return_net'] < 0]['position_change'].sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    cum_returns = np.exp(backtest_df['strategy_return_net'].cumsum()) - 1
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / (1 + running_max)
    max_drawdown = drawdown.min()
    
    risk_free_rate = 0.02 / 365 / 24 / 60
    returns_mean = backtest_df['strategy_return_net'].mean()
    returns_std = backtest_df['strategy_return_net'].std()
    sharpe_ratio = np.sqrt(365 * 24 * 60) * (returns_mean - risk_free_rate) / returns_std if returns_std > 0 else 0
    
    final_return = backtest_df['cum_strategy_return'].iloc[-1]
    market_return = backtest_df['cum_market_return'].iloc[-1]
    
    print(f"Backtest Results:")
    print(f"Total Return: {final_return:.2%}")
    print(f"Market Return: {market_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Total Trades: {int(total_trades)}")
    
    return backtest_df, {
        'total_return': final_return,
        'market_return': market_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': total_trades
    }
    
print('BACKTEST DONE')

'''
VISUALIZATION
def visualize_results(backtest_df, backtest_metrics):
    print("Inside visualize_results, type of backtest_df:", type(backtest_df))
    if not isinstance(backtest_df['final_return'], pd.DataFrame):
        raise TypeError(f"backtest_df should be a DataFrame, got {type(backtest_df)}")
    if 'final_return' not in backtest_df.columns:
        raise KeyError("'final_return' not found in backtest_df. Available columns: " + str(backtest_df.columns.tolist()))
    plt.plot(backtest_df.index, backtest_df['final_return'], label='Strategy')
    plt.title('Backtest Strategy Returns')
    plt.xlabel('Time')
    plt.ylabel('Final Return')
    plt.legend()
    plt.show()
    
    plt.close('all')
    gc.collect()
'''


'''
TRADING WORKFLOW
'''
def run_trading_workflow(file_path, target_horizon=5, sample_rate=0.25):
    df = pd.read_csv(file_path).sample(frac=sample_rate, random_state=42)
    df_featured = engineer_features(df)
    df_with_targets = create_targets(df_featured, horizons=[target_horizon])
    
    target_col = f'target_{target_horizon}'
    df_filtered, selected_features = select_features(df_with_targets, target_col)
    
    best_model_name, best_metrics, scaler = train_and_evaluate_models(df_filtered, selected_features, target_col, target_horizon)
    backtest_results = backtest_strategy(df_filtered, best_model_name, selected_features, target_col, scaler)
    #visualize_results(backtest_results, best_metrics)
    
    # Save artifacts
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('model.pkl', 'wb') as f:
        pickle.dump(best_model_name, f)
    
    return {
        'features': selected_features,
        'model': best_model_name,
        'scaler': scaler,
        'backtest_results': backtest_results
    }
print('WORKFLOW DONE')



if __name__ == "__main__":
    result = run_trading_workflow('BTC_1min.csv')