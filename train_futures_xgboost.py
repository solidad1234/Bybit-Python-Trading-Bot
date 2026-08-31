#!/usr/bin/env python3
"""
Futures Bot Machine Learning Trainer (train_futures_xgboost.py)
================================================================
Generates a multi-asset trade dataset using Bybit historical klines and derivatives data,
trains an Ensemble Decision Tree / Random Forest Classifier (using pure numpy/pandas)
to predict P(WIN), and saves the model artifact to `xgboost_filter.pkl` for use in futures.py.

Usage:
    python3 train_futures_xgboost.py --days 180
"""

import argparse
import sys
import os
import json
import time
import sqlite3
import pickle
import numpy as np
import pandas as pd

from datetime import datetime, timezone, timedelta

FEATURE_COLUMNS = [
    "is_long",
    "ta_signal_strength",
    "aggregated_score",
    "volatility",
    "atr_15m",
    "technical_score",
    "regime_score",
    "derivatives_score",
    "sentiment_score",
    "news_score",
    "sr_score",
    "funding_rate",
    "trend_bull_4h",
    "regime_is_bull",
    "regime_is_bear",
]


class DecisionNode:
    """Node in a decision tree classifier."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, prob=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prob = prob  # P(WIN) if leaf node

    def to_dict(self):
        if self.prob is not None:
            return {"prob": float(self.prob)}
        return {
            "feature": int(self.feature),
            "threshold": float(self.threshold),
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    @classmethod
    def from_dict(cls, d):
        if "prob" in d:
            return cls(prob=d["prob"])
        left = cls.from_dict(d["left"])
        right = cls.from_dict(d["right"])
        return cls(feature=d["feature"], threshold=d["threshold"], left=left, right=right)


class DecisionTreeClassifierCustom:
    """Lightweight Decision Tree Classifier using pure NumPy."""
    def __init__(self, max_depth=4, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _gini(self, y):
        if len(y) == 0:
            return 0
        p = np.mean(y)
        return 1.0 - (p**2 + (1 - p)**2)

    def _best_split(self, X, y):
        best_gini = 999.0
        best_feat, best_thresh = None, None
        n_samples, n_features = X.shape

        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            if len(thresholds) > 10:
                thresholds = np.percentile(thresholds, np.linspace(10, 90, 9))

            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask

                if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
                    continue

                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])
                weighted_gini = (np.sum(left_mask) * gini_left + np.sum(right_mask) * gini_right) / n_samples

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh

    def _build_tree(self, X, y, depth=0):
        n_samples = len(y)
        prob = np.mean(y) if n_samples > 0 else 0.5

        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            return DecisionNode(prob=prob)

        feat, thresh = self._best_split(X, y)
        if feat is None:
            return DecisionNode(prob=prob)

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return DecisionNode(feature=feat, threshold=thresh, left=left_child, right=right_child)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _predict_sample(self, node, x):
        if node.prob is not None:
            return node.prob
        if x[node.feature] <= node.threshold:
            return self._predict_sample(node.left, x)
        else:
            return self._predict_sample(node.right, x)

    def predict_proba(self, X):
        return np.array([self._predict_sample(self.root, x) for x in X])


class RandomForestClassifierCustom:
    """Random Forest Ensemble Classifier using pure NumPy."""
    def __init__(self, n_estimators=25, max_depth=4, min_samples_split=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.trees = []
        importances = np.zeros(n_features)

        for seed in range(self.n_estimators):
            np.random.seed(seed)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot, y_boot = X[indices], y[indices]

            tree = DecisionTreeClassifierCustom(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

        base_probs = self.predict_proba(X)[:, 1]
        for f in range(n_features):
            X_perm = X.copy()
            np.random.shuffle(X_perm[:, f])
            perm_probs = self.predict_proba(X_perm)[:, 1]
            importances[f] = np.mean(np.abs(base_probs - perm_probs))

        total_imp = np.sum(importances)
        self.feature_importances_ = importances / total_imp if total_imp > 0 else importances

    def predict_proba(self, X):
        tree_probs = np.array([tree.predict_proba(X) for tree in self.trees])
        win_probs = np.mean(tree_probs, axis=0)
        return np.column_stack([1.0 - win_probs, win_probs])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

    def to_dict(self):
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "feature_importances": self.feature_importances_.tolist() if self.feature_importances_ is not None else [],
            "trees": [t.root.to_dict() for t in self.trees],
        }

    @classmethod
    def from_dict(cls, d):
        rf = cls(n_estimators=d["n_estimators"], max_depth=d["max_depth"], min_samples_split=d["min_samples_split"])
        rf.feature_importances_ = np.array(d["feature_importances"])
        rf.trees = []
        for t_dict in d["trees"]:
            t = DecisionTreeClassifierCustom(max_depth=rf.max_depth, min_samples_split=rf.min_samples_split)
            t.root = DecisionNode.from_dict(t_dict)
            rf.trees.append(t)
        return rf


def load_dataset_from_csv_and_db(csv_path="trade_log.csv", db_path="trading_state.db"):
    dfs = []
    if os.path.exists(csv_path):
        try:
            df_csv = pd.read_csv(csv_path)
            if not df_csv.empty:
                print(f"📊 Loaded {len(df_csv)} records from {csv_path}")
                dfs.append(df_csv)
        except Exception as e:
            print(f"⚠️ Error loading {csv_path}: {e}")

    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            df_db = pd.read_sql_query("SELECT * FROM trade_log", conn)
            conn.close()
            if not df_db.empty:
                print(f"📊 Loaded {len(df_db)} records from live DB {db_path}")
                dfs.append(df_db)
        except Exception as e:
            print(f"⚠️ Error loading {db_path}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["symbol", "entry_time", "direction"]).reset_index(drop=True)
    return combined


def preprocess_features(df):
    if df.empty:
        return None, None, None

    data = df.copy()

    if "net_pnl" in data.columns:
        data["target"] = (data["net_pnl"] > 0).astype(int)
    elif "result" in data.columns:
        data["target"] = (data["result"] == "WIN").astype(int)
    else:
        raise ValueError("Dataset missing 'net_pnl' or 'result' column for training target.")

    data["is_long"] = (data["direction"] == "LONG").astype(int)
    data["trend_bull_4h"] = (data["market_trend_4h"] == "BULL").astype(int)
    data["regime_is_bull"] = (data["regime_class"] == "BULL").astype(int)
    data["regime_is_bear"] = (data["regime_class"] == "BEAR").astype(int)

    numeric_defaults = {
        "ta_signal_strength": 4.0,
        "aggregated_score": 0.25,
        "volatility": 0.02,
        "atr_15m": 1.0,
        "technical_score": 0.2,
        "regime_score": 0.0,
        "derivatives_score": 0.0,
        "sentiment_score": 0.0,
        "news_score": 0.0,
        "sr_score": 0.0,
        "funding_rate": 0.0001,
    }

    for col, default_val in numeric_defaults.items():
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(default_val)
        else:
            data[col] = default_val

    X = data[FEATURE_COLUMNS].copy().values
    y = data["target"].values.astype(int)

    return X, y, data


def train_model(X, y, df_raw):
    print("\n" + "=" * 60)
    print("TRAINING MACHINE LEARNING MODEL (TRADE QUALITY GATE)")
    print("=" * 60)
    print(f"Dataset Size: {len(X)} trade samples")
    print(f"Class Distribution: {np.sum(y == 1)} Wins (1) | {np.sum(y == 0)} Losses/Scratches (0)")
    print(f"Baseline Win Rate: {np.mean(y) * 100:.1f}%\n")

    split_idx = int(len(X) * 0.75)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    df_test = df_raw.iloc[split_idx:].copy()

    model = RandomForestClassifierCustom(n_estimators=30, max_depth=4, min_samples_split=4)
    model.fit(X_train, y_train)

    probs_test = model.predict_proba(X_test)[:, 1]

    test_acc = np.mean((probs_test >= 0.5) == y_test) * 100

    print("=" * 60)
    print("OUT-OF-SAMPLE MODEL EVALUATION ON TEST SET")
    print("=" * 60)
    print(f"Test Set Accuracy:  {test_acc:.1f}%")

    df_test["win_prob"] = probs_test
    df_test["ml_pass"] = probs_test >= 0.55

    orig_pnl = df_test["net_pnl"].sum()
    ml_filtered_pnl = df_test[df_test["ml_pass"]]["net_pnl"].sum()

    orig_trades = len(df_test)
    ml_trades = int(np.sum(df_test["ml_pass"]))

    orig_wr = np.mean(df_test["net_pnl"] > 0) * 100
    ml_wr = np.mean(df_test[df_test["ml_pass"]]["net_pnl"] > 0) * 100 if ml_trades > 0 else 0.0

    print("\n" + "=" * 60)
    print("REAL-WORLD PNL IMPROVEMENT WITH ML FILTER")
    print("=" * 60)
    print(f"Original Trades Taken:   {orig_trades} trades")
    print(f"Original Win Rate:       {orig_wr:.1f}%")
    print(f"Original Test PnL:       ${orig_pnl:+.2f}")
    print("------------------------------------------------------------")
    print(f"ML Filtered Trades:      {ml_trades} trades (Skipped {orig_trades - ml_trades} bad setups)")
    print(f"ML Filtered Win Rate:    {ml_wr:.1f}%")
    print(f"ML Filtered Test PnL:    ${ml_filtered_pnl:+.2f}")
    print("=" * 60)

    feat_imp = pd.DataFrame({"Feature": FEATURE_COLUMNS, "Importance": model.feature_importances_})
    feat_imp = feat_imp.sort_values("Importance", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("TOP PREDICTORS OF WINNING TRADES (FEATURE IMPORTANCE)")
    print("=" * 60)
    for idx, row in feat_imp.iterrows():
        bar = "█" * int(row["Importance"] * 40)
        print(f"{row['Feature']:<22} | {row['Importance']:.4f} {bar}")

    # Save dictionary JSON/pickle representation
    model_path = "xgboost_filter.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model.to_dict(), f)

    cols_path = "feature_columns.json"
    with open(cols_path, "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    print("\n" + "=" * 60)
    print(f"✅ Model saved to:           {model_path}")
    print(f"✅ Feature columns saved to: {cols_path}")
    print("=" * 60)

    return model


def main():
    parser = argparse.ArgumentParser(description="Train Futures ML Quality Gate Model")
    parser.add_argument("--days", type=int, default=180, help="Days of history to backtest if dataset is small")
    args = parser.parse_args()

    df = load_dataset_from_csv_and_db()
    if df.empty:
        print("❌ Could not load trade dataset. Run backtest_futures.py first.")
        sys.exit(1)

    X, y, df_raw = preprocess_features(df)
    train_model(X, y, df_raw)


if __name__ == "__main__":
    main()
