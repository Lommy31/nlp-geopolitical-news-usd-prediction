"""Task 2: feature extraction, baseline and combined model training/evaluation.

Methodology follows the validated approach from the team's baseline notebook
(technical FX features vs. technical + NLP sentiment/topic features, chronological
split, classical models only per the task's no-transformer constraint), applied
to the CNBC corpus.
"""

import re
from bisect import bisect_left

import numpy as np
import pandas as pd
import pysentiment2 as ps
import xgboost as xgb
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

FX_PATH = "data/processed/usd_idr_jisdor_clean.csv"
NEWS_RAW_PATH = "data/raw/news_cnbc.csv"

MIN_WORDS = 150
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

TECHNICAL_FEATURES = [
    "return_t", "return_lag1", "return_lag2", "return_lag3", "return_lag5",
    "volatility_5d", "volatility_20d", "ma_ratio_5d", "ma_ratio_20d",
    "momentum_5d", "momentum_10d", "rsi_14",
]
TITLE_SENTIMENT_FEATURES = [
    "vader_title_compound", "vader_title_pos", "vader_title_neg",
    "lm_title_polarity", "lm_title_subjectivity",
    "title_sentiment_decay3", "title_sentiment_decay5",
    "title_shock_extremity", "lm_title_shock", "sent_vol_interact",
]
BODY_SENTIMENT_FEATURES = [
    "vader_body_compound", "vader_body_pos", "vader_body_neg",
    "lm_body_polarity", "lm_body_subjectivity",
]
ACTIVITY_FEATURES = ["news_article_count", "has_news"]
TFIDF_FEATURES = [f"tfidf_topic_{i+1}" for i in range(5)]
ALL_COMBINED_FEATURES = (
    TECHNICAL_FEATURES + TITLE_SENTIMENT_FEATURES + BODY_SENTIMENT_FEATURES
    + ACTIVITY_FEATURES + TFIDF_FEATURES
)


# ---------------------------------------------------------------------------
# News cleaning
# ---------------------------------------------------------------------------

def clean_text_field(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_title(title):
    title = re.sub(r"^[^|]{1,20}\|\s*", "", str(title))
    title = title.lower()
    title = re.sub(r"[^a-z0-9\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return " ".join(title.split()[:6])


def clean_news(df_raw):
    df = df_raw.copy()
    df["Title"] = df["Title"].apply(clean_text_field)
    df["Text_clean"] = df["Text"].apply(clean_text_field)

    df["word_count"] = df["Text_clean"].str.split().str.len()
    df = df[df["word_count"] >= MIN_WORDS]

    df["title_key"] = df["Title"].apply(normalize_title)
    df = df.sort_values("word_count", ascending=False).drop_duplicates(subset="title_key")
    return df.drop(columns=["word_count", "title_key", "Text"])


def align_to_trading_days(df_news, fx_path=FX_PATH):
    trading_days = sorted(pd.read_csv(fx_path, parse_dates=["Date"])["Date"].dt.date.tolist())

    def align(d):
        idx = bisect_left(trading_days, d)
        return trading_days[idx] if idx < len(trading_days) else None

    df = df_news.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["Aligned_Date"] = df["Date"].apply(align)
    df = df.dropna(subset=["Aligned_Date"])
    df["Date"] = pd.to_datetime(df["Date"])
    df["Aligned_Date"] = pd.to_datetime(df["Aligned_Date"])
    return df


# ---------------------------------------------------------------------------
# Exchange rate feature engineering
# ---------------------------------------------------------------------------

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def build_rate_features(df_rates):
    d = df_rates.sort_values("Date").reset_index(drop=True).copy()
    d["return_t"] = d["USD_IDR"].pct_change()
    d["return_lag1"] = d["return_t"].shift(1)
    d["return_lag2"] = d["return_t"].shift(2)
    d["return_lag3"] = d["return_t"].shift(3)
    d["return_lag5"] = d["return_t"].shift(5)
    d["volatility_5d"] = d["return_t"].rolling(5).std()
    d["volatility_20d"] = d["return_t"].rolling(20).std()
    d["ma_ratio_5d"] = d["USD_IDR"] / d["USD_IDR"].rolling(5).mean()
    d["ma_ratio_20d"] = d["USD_IDR"] / d["USD_IDR"].rolling(20).mean()
    d["momentum_5d"] = (d["USD_IDR"] - d["USD_IDR"].shift(5)) / d["USD_IDR"].shift(5)
    d["momentum_10d"] = (d["USD_IDR"] - d["USD_IDR"].shift(10)) / d["USD_IDR"].shift(10)
    d["rsi_14"] = compute_rsi(d["USD_IDR"])
    d["next_close"] = d["USD_IDR"].shift(-1)
    d["next_return"] = (d["next_close"] - d["USD_IDR"]) / d["USD_IDR"]
    d["target_direction"] = (d["next_return"] > 0).astype(int)
    d = d.dropna().reset_index(drop=True)
    d["Date"] = d["Date"].dt.strftime("%Y-%m-%d")
    return d


# ---------------------------------------------------------------------------
# News sentiment feature engineering
# ---------------------------------------------------------------------------

def score_news_sentiment(df_news):
    d = df_news.copy()
    vader = SentimentIntensityAnalyzer()
    lm = ps.LM()

    v_title = d["Title"].apply(vader.polarity_scores)
    d["vader_title_compound"] = [s["compound"] for s in v_title]
    d["vader_title_pos"] = [s["pos"] for s in v_title]
    d["vader_title_neg"] = [s["neg"] for s in v_title]

    v_body = d["Text_clean"].apply(lambda t: vader.polarity_scores(t[:2500]))
    d["vader_body_compound"] = [s["compound"] for s in v_body]
    d["vader_body_pos"] = [s["pos"] for s in v_body]
    d["vader_body_neg"] = [s["neg"] for s in v_body]

    def lm_scores(text):
        try:
            s = lm.get_score(lm.tokenize(text))
            return float(s["Polarity"]), float(s["Subjectivity"])
        except Exception:
            return 0.0, 0.0

    lm_t = d["Title"].apply(lm_scores)
    d["lm_title_polarity"] = [s[0] for s in lm_t]
    d["lm_title_subjectivity"] = [s[1] for s in lm_t]

    lm_b = d["Text_clean"].apply(lambda t: lm_scores(t[:3000]))
    d["lm_body_polarity"] = [s[0] for s in lm_b]
    d["lm_body_subjectivity"] = [s[1] for s in lm_b]

    d["Aligned_Date_str"] = d["Aligned_Date"].dt.strftime("%Y-%m-%d")
    return d


def aggregate_and_merge(df_rates_clean, df_news_scored):
    agg_rules = {
        "vader_title_compound": ["mean", "min", "max"],
        "vader_title_pos": "mean",
        "vader_title_neg": "mean",
        "vader_body_compound": ["mean", "min", "max"],
        "vader_body_pos": "mean",
        "vader_body_neg": "mean",
        "lm_title_polarity": ["mean", "min", "max"],
        "lm_title_subjectivity": "mean",
        "lm_body_polarity": ["mean", "min", "max"],
        "lm_body_subjectivity": "mean",
    }
    daily = df_news_scored.groupby("Aligned_Date_str").agg(agg_rules)
    daily.columns = [f"{c}_{s}" if s != "mean" else c for c, s in daily.columns]
    daily["news_article_count"] = df_news_scored.groupby("Aligned_Date_str").size()
    daily["daily_titles"] = df_news_scored.groupby("Aligned_Date_str")["Title"].apply(lambda t: " ".join(t))
    daily = daily.reset_index().rename(columns={"Aligned_Date_str": "Date"})

    merged = pd.merge(df_rates_clean, daily, on="Date", how="left")
    merged["has_news"] = (~merged["news_article_count"].isna()).astype(int)

    nlp_cols = [c for c in daily.columns if c not in ("Date", "daily_titles")]
    for c in nlp_cols:
        merged[c] = merged[c].fillna(0.0)
    merged["daily_titles"] = merged["daily_titles"].fillna("")

    merged["title_sentiment_decay3"] = merged["vader_title_compound"].ewm(span=3, adjust=False).mean()
    merged["title_sentiment_decay5"] = merged["vader_title_compound"].ewm(span=5, adjust=False).mean()
    merged["title_shock_extremity"] = merged["vader_title_compound"].abs()
    merged["lm_title_shock"] = merged["lm_title_polarity"].abs()
    merged["sent_vol_interact"] = merged["vader_title_compound"] * merged["volatility_5d"]
    return merged


# ---------------------------------------------------------------------------
# Splitting and TF-IDF topics
# ---------------------------------------------------------------------------

def chronological_split(merged, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    n = len(merged)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_df = merged.iloc[:n_train].reset_index(drop=True)
    val_df = merged.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = merged.iloc[n_train + n_val:].reset_index(drop=True)
    return train_df, val_df, test_df


def add_tfidf_topics(train_df, val_df, test_df, n_components=5):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=500, stop_words="english", sublinear_tf=True)
    svd = TruncatedSVD(n_components=n_components, random_state=42)

    svd_train = svd.fit_transform(tfidf.fit_transform(train_df["daily_titles"]))
    svd_val = svd.transform(tfidf.transform(val_df["daily_titles"]))
    svd_test = svd.transform(tfidf.transform(test_df["daily_titles"]))

    for i in range(n_components):
        col = f"tfidf_topic_{i+1}"
        train_df[col] = svd_train[:, i]
        val_df[col] = svd_val[:, i]
        test_df[col] = svd_test[:, i]
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Modeling and evaluation
# ---------------------------------------------------------------------------

XGB_PARAMS = dict(n_estimators=70, max_depth=2, learning_rate=0.03, subsample=0.75,
                  colsample_bytree=0.75, reg_alpha=0.5, reg_lambda=2.0, random_state=42)
RF_PARAMS = dict(n_estimators=100, max_depth=3, min_samples_leaf=4, random_state=42)


def evaluate_model(y_true, y_pred, y_prob=None, name="Model"):
    return {
        "Model": name,
        "Directional Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 (Macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob) if y_prob is not None else 0.50,
    }


def train_baseline_models(train_df, test_df, feature_cols=TECHNICAL_FEATURES):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])
    y_train = train_df["target_direction"].values
    y_test = test_df["target_direction"].values

    results = []
    maj = int(np.round(np.mean(y_train)))
    results.append(evaluate_model(y_test, np.full_like(y_test, maj), name="Naive: Majority Class"))
    results.append(evaluate_model(y_test, (test_df["return_t"].values > 0).astype(int),
                                  name="Naive: Momentum (t-1)"))

    models = {
        "TS Baseline: Logistic Regression": LogisticRegression(C=0.05, max_iter=1000, random_state=42),
        "TS Baseline: XGBoost": xgb.XGBClassifier(**XGB_PARAMS),
        "TS Baseline: Random Forest": RandomForestClassifier(**RF_PARAMS),
    }
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results.append(evaluate_model(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1], name))
        fitted[name] = model

    return pd.DataFrame(results), fitted, scaler


def train_combined_models(train_df, test_df, feature_cols=ALL_COMBINED_FEATURES):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])
    y_train = train_df["target_direction"].values
    y_test = test_df["target_direction"].values

    lr = LogisticRegression(C=0.05, penalty="l2", max_iter=1000, random_state=42).fit(X_train, y_train)
    rf = RandomForestClassifier(**RF_PARAMS).fit(X_train, y_train)
    xgb_model = xgb.XGBClassifier(**XGB_PARAMS).fit(X_train, y_train)
    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("xgb", xgb_model)], voting="soft"
    ).fit(X_train, y_train)

    fitted = {"Combined: Logistic Regression": lr, "Combined: Random Forest": rf,
             "Combined: XGBoost": xgb_model, "Combined: Soft-Voting Ensemble": ensemble}
    results = [evaluate_model(y_test, m.predict(X_test), m.predict_proba(X_test)[:, 1], name)
              for name, m in fitted.items()]
    return pd.DataFrame(results), fitted, scaler


def evaluate_on_active_days(test_df, fitted_baseline, scaler_baseline, fitted_combined, scaler_combined,
                            feature_cols_baseline=TECHNICAL_FEATURES, feature_cols_combined=ALL_COMBINED_FEATURES):
    active = test_df[test_df["has_news"] == 1]
    if len(active) < 10 or active["target_direction"].nunique() < 2:
        return pd.DataFrame(), active

    y_active = active["target_direction"].values
    Xa_base = scaler_baseline.transform(active[feature_cols_baseline])
    Xa_comb = scaler_combined.transform(active[feature_cols_combined])

    results = []
    for name in ("TS Baseline: Random Forest", "TS Baseline: XGBoost"):
        m = fitted_baseline[name]
        results.append(evaluate_model(y_active, m.predict(Xa_base), m.predict_proba(Xa_base)[:, 1], name))
    for name in ("Combined: Random Forest", "Combined: XGBoost", "Combined: Soft-Voting Ensemble"):
        m = fitted_combined[name]
        results.append(evaluate_model(y_active, m.predict(Xa_comb), m.predict_proba(Xa_comb)[:, 1], name))
    return pd.DataFrame(results), active


def train_regression(train_df, test_df, feature_cols=ALL_COMBINED_FEATURES):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])
    y_train = train_df["next_return"].values
    y_test = test_df["next_return"].values

    ridge = Ridge(alpha=10.0, random_state=42).fit(X_train, y_train)
    pred_r = ridge.predict(X_test)
    xgb_reg = xgb.XGBRegressor(n_estimators=50, max_depth=2, learning_rate=0.03, random_state=42)
    xgb_reg.fit(X_train, y_train)
    pred_x = xgb_reg.predict(X_test)

    return pd.DataFrame([
        {"Model": "Ridge", "RMSE": np.sqrt(mean_squared_error(y_test, pred_r)),
         "MAE": mean_absolute_error(y_test, pred_r),
         "Directional Accuracy": np.mean((pred_r > 0) == (y_test > 0))},
        {"Model": "XGBoost", "RMSE": np.sqrt(mean_squared_error(y_test, pred_x)),
         "MAE": mean_absolute_error(y_test, pred_x),
         "Directional Accuracy": np.mean((pred_x > 0) == (y_test > 0))},
    ])


def ablation_study(train_df, test_df):
    configs = {
        "1. Technical Only": TECHNICAL_FEATURES,
        "2. Technical + Title Sentiment": TECHNICAL_FEATURES + TITLE_SENTIMENT_FEATURES,
        "3. Technical + Body Sentiment": TECHNICAL_FEATURES + BODY_SENTIMENT_FEATURES,
        "4. Technical + Title + Body": TECHNICAL_FEATURES + TITLE_SENTIMENT_FEATURES + BODY_SENTIMENT_FEATURES,
        "5. Full Multimodal": ALL_COMBINED_FEATURES,
    }
    y_train = train_df["target_direction"].values
    y_test = test_df["target_direction"].values

    rows = []
    for name, feats in configs.items():
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[feats])
        X_test = scaler.transform(test_df[feats])
        model = xgb.XGBClassifier(**XGB_PARAMS).fit(X_train, y_train)
        m = evaluate_model(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1], name)
        m["Num Features"] = len(feats)
        rows.append(m)
    return pd.DataFrame(rows)[["Model", "Num Features", "Directional Accuracy", "F1 (Macro)", "ROC-AUC"]]
