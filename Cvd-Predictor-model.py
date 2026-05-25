"""
Hybrid XGBoost + Neural Network model for wearable CVD risk prediction.

Dataset: wearable_cvd_synthetic_dataset.xlsx (150 participants × 30 days).

Architecture
------------
              ┌────────────────────────┐
  past-7-day  │   1D-CNN encoder       │     8-d
  vitals  ──► │   (PyTorch)            │ ──► embedding ─┐
  (7×9)       └────────────────────────┘                │
                                                        ▼
  statics + current-day + 7-d rolling ────► concat ──► XGBoost ──► P(alert)

Target: physician_review_alert (binary, ~3% positive prevalence).
Split:  participant-level (prevents same-person leakage across train/test).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import (average_precision_score, classification_report,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = Path("/mnt/user-data/uploads/wearable_cvd_synthetic_dataset.xlsx")
SEQ_LEN = 7          # days of history fed to the CNN
EMBED_DIM = 4        # small embedding → summarises, doesn't crowd out tabular signal
NN_EPOCHS = 60       # upper bound; early stopping decides actual epoch count
NN_PATIENCE = 8
NN_BATCH = 128
NN_LR = 1e-3
RANDOM_STATE = 42

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load and merge
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(path: Path) -> pd.DataFrame:
    """Join the three relevant sheets into one (participant_id, date) row table."""
    sheets = pd.read_excel(path, sheet_name=["Participants",
                                             "Daily_Wearable_Data",
                                             "Daily_Risk_Scores"])
    participants = sheets["Participants"]
    daily = sheets["Daily_Wearable_Data"]
    risk = sheets["Daily_Risk_Scores"]

    df = (daily
          .merge(risk, on=["participant_id", "date"], how="inner")
          .merge(participants, on="participant_id", how="left")
          .sort_values(["participant_id", "date"])
          .reset_index(drop=True))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature engineering
# ─────────────────────────────────────────────────────────────────────────────
# Channels fed into the 1D-CNN — multivariate daily time series.
SEQ_CHANNELS = ["resting_hr_bpm", "mean_hr_bpm", "max_hr_bpm",
                "hrv_sdnn_ms", "sleep_duration_h", "sleep_efficiency_pct",
                "spo2_nocturnal_min_pct", "odi3_events_per_hour", "step_count"]

# Categorical statics — one-hot encoded for the tabular block.
CAT_COLS = ["sex", "smoking_status", "skin_tone_fitzpatrick",
            "lifestyle_profile", "occupation_category", "device_model"]

# Numeric tabular features (statics + same-day vitals + 7-day rolling stats).
NUM_COLS = [
    # statics
    "age_years", "altitude_residence_m", "bmi",
    "hypertension_dx", "diabetes_dx", "osa_dx", "family_history_cvd",
    "total_cholesterol_mgdl", "hdl_mgdl",
    # same-day wearable
    "device_worn_hours", "step_count", "active_minutes",
    "resting_hr_bpm", "mean_hr_bpm", "max_hr_bpm", "hrv_sdnn_ms",
    "sleep_duration_h", "sleep_efficiency_pct",
    "spo2_mean_pct", "spo2_nocturnal_min_pct", "odi3_events_per_hour",
    "vo2_max_estimate", "ppg_signal_quality_pct",
    # 7-day rolling (already in Daily_Risk_Scores)
    "roll7_resting_hr", "roll7_hrv_sdnn", "roll7_steps",
    "roll7_sleep_h", "roll7_spo2_nadir", "roll7_odi",
]


def build_sequence_tensor(df: pd.DataFrame, seq_len: int = SEQ_LEN) -> np.ndarray:
    """
    For each row, gather the previous `seq_len` days of SEQ_CHANNELS for the
    same participant. Days that fall off the start of the window are
    forward-filled from the earliest available day (avoids look-ahead).
    Returns array of shape (n_rows, seq_len, n_channels).
    """
    n_rows = len(df)
    n_ch = len(SEQ_CHANNELS)
    seqs = np.zeros((n_rows, seq_len, n_ch), dtype=np.float32)

    for _, group in df.groupby("participant_id", sort=False):
        idx = group.index.to_numpy()
        vals = group[SEQ_CHANNELS].to_numpy(dtype=np.float32)
        for i in range(len(idx)):
            start = max(0, i - seq_len + 1)
            window = vals[start:i + 1]
            if len(window) < seq_len:                  # pad at the front
                pad = np.repeat(window[:1], seq_len - len(window), axis=0)
                window = np.vstack([pad, window])
            seqs[idx[i]] = window
    return seqs


def build_tabular(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode categoricals + return the tabular feature matrix."""
    cat = pd.get_dummies(df[CAT_COLS].astype(str), drop_first=False)
    num = df[NUM_COLS].copy()
    X = pd.concat([num, cat], axis=1)
    X = X.fillna(X.median(numeric_only=True))
    return X, list(X.columns)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Neural network: 1D-CNN encoder + classification head
# ─────────────────────────────────────────────────────────────────────────────
class SequenceEncoder(nn.Module):
    """
    Small 1D-CNN: takes (B, n_channels, seq_len) → embedding (B, EMBED_DIM).
    A linear classification head sits on top of the embedding during training;
    once trained we drop the head and use the embedding as features for XGBoost.
    """
    def __init__(self, n_channels: int, seq_len: int, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                   # → (B, 32, 1)
        )
        self.embed = nn.Linear(32, embed_dim)
        self.head = nn.Linear(embed_dim, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).squeeze(-1)                   # (B, 32)
        return torch.relu(self.embed(h))               # (B, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x)).squeeze(-1)   # logits


def train_nn(X_seq: np.ndarray, y: np.ndarray, train_idx: np.ndarray,
             pos_weight: float, participants: np.ndarray) -> SequenceEncoder:
    """
    End-to-end CNN training on the alert target, with early stopping on a
    participant-disjoint validation slice carved out of the training set.
    """
    # Carve a participant-level validation slice (~20% of train participants).
    train_p = np.unique(participants[train_idx])
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(train_p)
    val_p = set(train_p[:max(1, len(train_p) // 5)])
    val_mask = np.isin(participants[train_idx], list(val_p))
    tr_sub_idx = train_idx[~val_mask]
    val_sub_idx = train_idx[val_mask]

    def make_loader(idx, shuffle):
        xt = torch.from_numpy(X_seq[idx]).permute(0, 2, 1)
        yt = torch.from_numpy(y[idx]).float()
        return DataLoader(TensorDataset(xt, yt), batch_size=NN_BATCH, shuffle=shuffle)

    train_loader = make_loader(tr_sub_idx, shuffle=True)
    val_loader = make_loader(val_sub_idx, shuffle=False)

    model = SequenceEncoder(n_channels=len(SEQ_CHANNELS), seq_len=SEQ_LEN)
    opt = torch.optim.Adam(model.parameters(), lr=NN_LR)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    best_val = float("inf")
    best_state = None
    stale = 0
    for epoch in range(NN_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = sum(loss_fn(model(xb), yb).item() * xb.size(0)
                           for xb, yb in val_loader) / len(val_sub_idx)
        if val_loss < best_val - 1e-4:
            best_val, best_state, stale = val_loss, {k: v.clone()
                                                     for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
        if (epoch + 1) % 10 == 0 or stale == 0:
            print(f"  epoch {epoch+1:2d}  val_loss={val_loss:.4f}"
                  f"{'  ✓ best' if stale == 0 else ''}")
        if stale >= NN_PATIENCE:
            print(f"  early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def extract_embeddings(model: SequenceEncoder, X_seq: np.ndarray) -> np.ndarray:
    """Run the encoder over every row and return the embedding matrix."""
    model.eval()
    xt = torch.from_numpy(X_seq).permute(0, 2, 1)
    emb = model.encode(xt).cpu().numpy()
    return emb


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────
def report(name: str, y_true: np.ndarray, p: np.ndarray) -> dict:
    auc = roc_auc_score(y_true, p)
    ap = average_precision_score(y_true, p)
    print(f"\n── {name} ──")
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"  PR-AUC  : {ap:.4f}")
    print(classification_report(y_true, (p >= 0.5).astype(int),
                                digits=3, zero_division=0))
    return {"name": name, "roc_auc": auc, "pr_auc": ap}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading dataset…")
    df = load_dataset(DATA_PATH)
    print(f"  merged rows: {len(df):,}   participants: {df['participant_id'].nunique()}")

    y = df["physician_review_alert"].to_numpy(dtype=np.float32)
    print(f"  positive class prevalence: {y.mean():.2%}")

    print("\nBuilding features…")
    # Impute NaN in sequence channels (e.g. SpO2 on devices that don't support it)
    # using each channel's median computed on the full table.
    df[SEQ_CHANNELS] = df[SEQ_CHANNELS].fillna(df[SEQ_CHANNELS].median(numeric_only=True))

    X_tab_df, tab_cols = build_tabular(df)
    X_tab = X_tab_df.to_numpy(dtype=np.float32)
    X_seq = build_sequence_tensor(df)
    print(f"  tabular : {X_tab.shape}   sequences : {X_seq.shape}")

    # Participant-level split — no person appears in both train and test.
    participants = df["participant_id"].unique()
    train_p, test_p = train_test_split(participants, test_size=0.25,
                                       random_state=RANDOM_STATE)
    train_idx = np.where(df["participant_id"].isin(train_p))[0]
    test_idx = np.where(df["participant_id"].isin(test_p))[0]
    print(f"  train rows: {len(train_idx)}   test rows: {len(test_idx)}")

    # Standardise sequence channels using training-only stats (no leakage).
    flat_train = X_seq[train_idx].reshape(-1, X_seq.shape[-1])
    seq_mean = flat_train.mean(axis=0)
    seq_std = flat_train.std(axis=0) + 1e-6
    X_seq = (X_seq - seq_mean) / seq_std

    # Scale tabular features (XGBoost is scale-invariant but the
    # NN-only baseline below benefits from it).
    scaler = StandardScaler().fit(X_tab[train_idx])
    X_tab_s = scaler.transform(X_tab)

    pos_weight = (y[train_idx] == 0).sum() / max(1, (y[train_idx] == 1).sum())
    print(f"  scale_pos_weight = {pos_weight:.1f}")

    # ── train the neural sequence encoder ──────────────────────────────────
    print("\nTraining 1D-CNN sequence encoder…")
    participants_arr = df["participant_id"].to_numpy()
    nn_model = train_nn(X_seq, y, train_idx,
                        pos_weight=pos_weight, participants=participants_arr)

    print("Extracting learned embeddings…")
    emb = extract_embeddings(nn_model, X_seq)
    print(f"  embedding matrix: {emb.shape}")

    # ── baseline A: XGBoost on tabular features only ───────────────────────
    print("\nTraining baseline XGBoost (tabular only)…")
    xgb_base = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=pos_weight, eval_metric="aucpr",
        random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
    )
    xgb_base.fit(X_tab[train_idx], y[train_idx],
                 eval_set=[(X_tab[test_idx], y[test_idx])], verbose=False)
    p_base = xgb_base.predict_proba(X_tab[test_idx])[:, 1]

    # ── baseline B: NN-only logits ─────────────────────────────────────────
    with torch.no_grad():
        nn_model.eval()
        logits = nn_model(torch.from_numpy(X_seq[test_idx]).permute(0, 2, 1))
        p_nn = torch.sigmoid(logits).cpu().numpy()

    # ── hybrid 1: stacking — XGBoost on [tabular ∪ NN embeddings] ──────────
    print("Training hybrid XGBoost (tabular + NN embeddings)…")
    X_hyb = np.hstack([X_tab, emb])
    xgb_hyb = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=pos_weight, eval_metric="aucpr",
        random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
    )
    xgb_hyb.fit(X_hyb[train_idx], y[train_idx],
                eval_set=[(X_hyb[test_idx], y[test_idx])], verbose=False)
    p_hyb = xgb_hyb.predict_proba(X_hyb[test_idx])[:, 1]

    # ── hybrid 2: probability ensemble (geometric-mean of NN and XGB) ──────
    # Often more robust than stacking on small / noisy data because each
    # branch keeps its own inductive bias.
    p_ens = np.sqrt(np.clip(p_base, 1e-6, 1) * np.clip(p_nn, 1e-6, 1))

    # ── results ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS (held-out participants)")
    print("=" * 60)
    rows = [
        report("Neural net only (1D-CNN)", y[test_idx], p_nn),
        report("XGBoost only (tabular)", y[test_idx], p_base),
        report("Hybrid 1: stacked (XGB on tabular + NN emb)", y[test_idx], p_hyb),
        report("Hybrid 2: probability ensemble (NN × XGB)", y[test_idx], p_ens),
    ]

    print("\nSummary")
    print("-" * 60)
    print(f"{'model':50s} {'ROC-AUC':>9s} {'PR-AUC':>9s}")
    for r in rows:
        print(f"{r['name']:50s} {r['roc_auc']:>9.4f} {r['pr_auc']:>9.4f}")

    # Top tabular features in the stacked hybrid model (sanity check).
    all_cols = tab_cols + [f"nn_emb_{i}" for i in range(EMBED_DIM)]
    imp = pd.Series(xgb_hyb.feature_importances_, index=all_cols)
    print("\nTop 15 features (stacked hybrid):")
    print(imp.sort_values(ascending=False).head(15).to_string())


if __name__ == "__main__":
    main()
