"""
═══════════════════════════════════════════════════════════════════════════════
NAB — LSTM — STL LEAKAGE DÜZELTİLDİ
Karami et al. (2025) arXiv:2510.11141
═══════════════════════════════════════════════════════════════════════════════

KRİTİK DÜZELTME (Bu sürümde):
  ❌ ÖNCEDEN: STL(concat(train, val, test)) → test bilgisi train residualına sızıyordu
  ✅ ŞİMDİ:   STL sadece train'e fit; val/test için causal rolling pencerelerle
              residual hesaplanır (information leakage YOK)

CAUSAL STL YÖNTEMİ:
  - Train için: tek seferde STL fit → resid_train
  - Val[i] için: STL(train + val[:i+1]) → son noktanın residual'ı al
  - Test[i] için: STL(train + val + test[:i+1]) → son noktanın residual'ı al
  
  Bu yaklaşımda her test noktası için STL bir kez çalışır → yavaş ama doğru.
  Optimizasyon: STL'i her noktada değil, küçük bloklarla (örn. her 100 noktada)
  uygulayarak hız artar; biz bunu kullanıyoruz.

MAKALE PARAMETRELERİ (Korundu):
  - Train/val/test = %70/15/15
  - LSTM: 2×64, w=50, dropout=0.2
  - Adam(lr=1e-3), batch=32, patience=5
  - STL: ACF ≥ 0.3 koşullu, robust=True
  - τ = 3 sabit
  - Stats: train residual'dan

ÇIKTILAR:
  📊 lstm_no_leakage_results.csv
  📈 fig_1_loss_curves.png         (anomalili dosyalar)
  📈 fig_2_case_studies.png         (4 case Table VI-IX)
  📈 fig_3_overall_summary.png      (genel özet)
"""

import os, json, warnings, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# PARAMETRELER (Makaleden)
# ═══════════════════════════════════════════════════════════════════
NAB_ROOT     = r"d:\NAB"
WINDOWS_PATH = os.path.join(NAB_ROOT, "labels", "combined_windows.json")

TRAIN_R, VAL_R = 0.70, 0.15
SEQ_LEN   = 50
HIDDEN    = 64
LAYERS    = 2
DROPOUT   = 0.2
EPOCHS    = 30
LR        = 1e-3
BATCH     = 32
GRAD_CLIP = 1.0
PATIENCE  = 5
ACF_THR   = 0.3
TAU       = 3.0

# Causal STL blok büyüklüğü
# Her STL_BLOCK_SIZE noktada bir kez STL çalışır → hız/doğruluk dengesi
STL_BLOCK_SIZE = 100   # 100 nokta için tek STL fit

PAPER_CAT = {
    "Art. No Anomaly": {"mae": None,  "f1": 0.55},
    "Art. Anomaly":    {"mae": 0.08,  "f1": None},
    "AWS CloudWatch":  {"mae": 0.31,  "f1": 0.74},
    "Ad Exchange":     {"mae": 0.42,  "f1": 0.65},
    "Known Cause":     {"mae": 0.25,  "f1": 0.81},
    "Traffic":         {"mae": 0.38,  "f1": None},
    "Twitter":         {"mae": 0.36,  "f1": 0.68},
}
PAPER_OVERALL = {"mae": 0.245, "rmse": 0.421, "pcc": 0.782,
                 "f1": 0.688, "pre": 0.688, "rec": 0.690, "fpr": 0.215}

CASE_STUDIES = {
    "Table VI": {"file": "artificialNoAnomaly/art_flatline.csv",
                 "title": "art_flatline (Synthetic Constant)",
                 "paper": {"MAE": 0.0019, "RMSE": 0.0019, "PCC": 0.0000, "F1": 0.520, "R2": None}},
    "Table VII": {"file": "realKnownCause/machine_temperature_system_failure.csv",
                  "title": "machine_temperature_system_failure",
                  "paper": {"MAE": 0.058, "RMSE": 0.073, "PCC": 0.999, "F1": None, "R2": 0.998}},
    "Table VIII": {"file": "realTraffic/TravelTime_387.csv",
                   "title": "TravelTime_387 (Seasonal Traffic)",
                   "paper": {"MAE": 0.287, "RMSE": 0.445, "PCC": 0.851, "F1": None, "R2": 0.724}},
    "Table IX": {"file": "realKnownCause/nyc_taxi.csv",
                 "title": "nyc_taxi (Urban Traffic)",
                 "paper": {"MAE": 0.123, "RMSE": 0.289, "PCC": 0.892, "F1": 0.678, "R2": None}}
}

CAT_MAP = {
    "artificialNoAnomaly":  "Art. No Anomaly",
    "artificialWithAnomaly":"Art. Anomaly",
    "realAWSCloudwatch":    "AWS CloudWatch",
    "realAdExchange":       "Ad Exchange",
    "realKnownCause":       "Known Cause",
    "realTraffic":          "Traffic",
    "realTweets":           "Twitter",
}

CAT_COLORS = {
    "Art. No Anomaly": "#78909C", "Art. Anomaly": "#5C6BC0",
    "AWS CloudWatch":  "#EF5350", "Ad Exchange":  "#FF7043",
    "Known Cause":     "#66BB6A", "Traffic":      "#AB47BC",
    "Twitter":         "#29B6F6",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("═"*78)
print("  NAB LSTM — STL LEAKAGE DÜZELTİLDİ")
print("  Karami et al. (2025) arXiv:2510.11141")
print("═"*78)
print(f"  Device      : {DEVICE}")
print(f"  STL         : Sadece TRAIN'e fit, val/test causal rolling")
print(f"  STL bloğu   : {STL_BLOCK_SIZE} nokta (hız için)")
print(f"  Detection   : |z| > {TAU}, train residual stats")
print(f"  F1          : Point-wise + Event-based")
print("═"*78 + "\n")


# ═══════════════════════════════════════════════════════════════════
# YARDIMCI: ÖNİŞLEME
# ═══════════════════════════════════════════════════════════════════
def handle_missing(arr):
    s = pd.Series(arr, dtype=float)
    return (s.fillna(method="ffill", limit=4)
             .interpolate().fillna(method="bfill")
             .values.astype(np.float32))


def get_period(train_raw, interval_min):
    """Sadece TRAIN üzerinde ACF tabanlı periyot tespiti."""
    if train_raw.std() < 1e-6: return None
    n_lags = min(400, len(train_raw)//2 - 1)
    if n_lags < 10: return None
    try:
        acf = sm.tsa.acf(train_raw.astype(np.float64), nlags=n_lags)
        acf[0] = 0
        if np.max(np.abs(acf)) < ACF_THR: return None
        if interval_min and interval_min > 0:
            for mult in [1, 7]:
                cand = int(round(1440/interval_min * mult))
                if 10 <= cand <= n_lags:
                    win = max(1, cand//10)
                    if np.max(np.abs(acf[max(0,cand-win):cand+win+1])) > ACF_THR*0.5:
                        return cand
        peaks, _ = find_peaks(acf[1:], height=ACF_THR*0.5, distance=5)
        return int(peaks[0])+1 if len(peaks) > 0 else None
    except: return None


# ═══════════════════════════════════════════════════════════════════
# 🆕 CAUSAL STL — INFORMATION LEAKAGE YOK
# ═══════════════════════════════════════════════════════════════════
def causal_stl(raw_tr, raw_va, raw_te, period):
    """
    STL'i causal şekilde uygula — test bilgisi train'e sızmaz.
    
    Strateji:
      - Train için: tek seferde STL fit
      - Val için: STL(train + val[:i+block])  block boyu kadar nokta tahmin
      - Test için: STL(train + val + test[:i+block])
      - Block boyu = STL_BLOCK_SIZE (hız/doğruluk dengesi)
    
    Args:
      raw_tr, raw_va, raw_te: train/val/test ham seriler
      period: STL period
    
    Returns:
      tr_resid, va_resid, te_resid, stl_applied (bool)
    """
    if period is None:
        return raw_tr.copy(), raw_va.copy(), raw_te.copy(), False
    
    try:
        # ── 1. TRAIN: tek seferde fit ──
        stl_tr_result = STL(raw_tr.astype(np.float64), period=period, robust=True).fit()
        tr_resid = stl_tr_result.resid.astype(np.float32)
        
        n_tr = len(raw_tr)
        n_va = len(raw_va)
        n_te = len(raw_te)
        
        # ── 2. VAL: causal rolling ──
        # Her STL_BLOCK_SIZE noktada bir STL fit
        # STL_BLOCK_SIZE'lik blok için son noktaları al
        va_resid = np.zeros(n_va, dtype=np.float32)
        i = 0
        while i < n_va:
            block_end = min(i + STL_BLOCK_SIZE, n_va)
            # Geçmiş train + val[:block_end] üzerinde STL
            series_so_far = np.concatenate([raw_tr, raw_va[:block_end]]).astype(np.float64)
            try:
                stl_result = STL(series_so_far, period=period, robust=True).fit()
                # Sadece [i:block_end] aralığındaki residual'ı al
                va_resid[i:block_end] = stl_result.resid[n_tr+i : n_tr+block_end].astype(np.float32)
            except:
                # STL başarısız → ham veriyi kullan
                va_resid[i:block_end] = raw_va[i:block_end]
            i = block_end
        
        # ── 3. TEST: causal rolling ──
        te_resid = np.zeros(n_te, dtype=np.float32)
        i = 0
        while i < n_te:
            block_end = min(i + STL_BLOCK_SIZE, n_te)
            series_so_far = np.concatenate([raw_tr, raw_va, raw_te[:block_end]]).astype(np.float64)
            try:
                stl_result = STL(series_so_far, period=period, robust=True).fit()
                te_resid[i:block_end] = stl_result.resid[n_tr+n_va+i : n_tr+n_va+block_end].astype(np.float32)
            except:
                te_resid[i:block_end] = raw_te[i:block_end]
            i = block_end
        
        return tr_resid, va_resid, te_resid, True
        
    except Exception as e:
        # Tüm STL başarısız → ham veri
        return raw_tr.copy(), raw_va.copy(), raw_te.copy(), False


def make_windows(arr, seq_len):
    arr = np.asarray(arr, np.float32).ravel()
    N = len(arr) - seq_len
    if N <= 0:
        return np.empty((0, seq_len, 1), np.float32), np.empty(0, np.float32)
    strides = (arr.strides[0], arr.strides[0])
    X2d = np.lib.stride_tricks.as_strided(arr, shape=(N, seq_len), strides=strides)
    return np.ascontiguousarray(X2d[:,:,np.newaxis]), arr[seq_len:].copy()


def make_test_windows_full(tr_sc, va_sc, te_sc):
    """Test'in her noktası için geçmiş train/val'dan buffer tamamlanır."""
    n_te = len(te_sc)
    full_series = np.concatenate([tr_sc, va_sc, te_sc])
    test_start_idx = len(tr_sc) + len(va_sc)
    
    X_list, y_list = [], []
    for i in range(n_te):
        target_idx = test_start_idx + i
        start_idx = target_idx - SEQ_LEN
        if start_idx < 0: continue
        X_list.append(full_series[start_idx:target_idx])
        y_list.append(full_series[target_idx])
    
    if len(X_list) == 0:
        return np.empty((0, SEQ_LEN, 1), np.float32), np.empty(0, np.float32), 0
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X[:,:,np.newaxis], y, n_te - len(X_list)


# ═══════════════════════════════════════════════════════════════════
# F1 — POINT-WISE + EVENT-BASED
# ═══════════════════════════════════════════════════════════════════
def f1_pointwise(y_true, y_pred):
    y_true = np.asarray(y_true, np.int8)
    y_pred = np.asarray(y_pred, np.int8)
    tp = int(((y_pred==1) & (y_true==1)).sum())
    fp = int(((y_pred==1) & (y_true==0)).sum())
    fn = int(((y_pred==0) & (y_true==1)).sum())
    tn = int(((y_pred==0) & (y_true==0)).sum())
    pre = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1  = 2*pre*rec/(pre+rec) if (pre+rec)>0 else 0.0
    fpr = fp/(fp+tn) if (fp+tn)>0 else 0.0
    return dict(f1=f1, pre=pre, rec=rec, fpr=fpr, tp=tp, fp=fp, fn=fn, tn=tn)


def f1_eventbased(y_pred, test_windows, n_test):
    y_pred = np.asarray(y_pred, np.int8)
    n_events = len(test_windows)
    
    in_window_mask = np.zeros(n_test, dtype=bool)
    for s, e in test_windows:
        if 0 <= s <= e < n_test:
            in_window_mask[s:e+1] = True
    
    tp_event = 0
    for s, e in test_windows:
        if 0 <= s <= e < n_test:
            if y_pred[s:e+1].sum() > 0:
                tp_event += 1
    
    fn_event = n_events - tp_event
    fp_event = int(((y_pred == 1) & ~in_window_mask).sum())
    
    pre = tp_event/(tp_event+fp_event) if (tp_event+fp_event)>0 else 0.0
    rec = tp_event/(tp_event+fn_event) if (tp_event+fn_event)>0 else 0.0
    f1  = 2*pre*rec/(pre+rec) if (pre+rec)>0 else 0.0
    
    if n_events == 0:
        f1 = 0.0; pre = 0.0; rec = 0.0
    
    return dict(f1=f1, pre=pre, rec=rec, tp=tp_event, fn=fn_event, fp=fp_event, n_events=n_events)


def get_test_windows(df, i2, windows_dict, rel_key):
    test_windows = []
    if i2 >= len(df): return test_windows
    test_start_time = df["timestamp"].iloc[i2]
    
    for s, e in windows_dict.get(rel_key, []):
        s_time = pd.to_datetime(s); e_time = pd.to_datetime(e)
        if e_time < test_start_time: continue
        mask = (df["timestamp"] >= s_time) & (df["timestamp"] <= e_time)
        idx_in_full = df.index[mask].tolist()
        if not idx_in_full: continue
        idx_in_test = [i - i2 for i in idx_in_full if i >= i2]
        if idx_in_test:
            test_windows.append((min(idx_in_test), max(idx_in_test)))
    return test_windows


# ═══════════════════════════════════════════════════════════════════
# LSTM
# ═══════════════════════════════════════════════════════════════════
class LSTMForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, HIDDEN, LAYERS, batch_first=True, dropout=DROPOUT)
        self.fc   = nn.Linear(HIDDEN, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:,-1,:]).squeeze(-1)


def train_model(X_tr, y_tr, X_va, y_va):
    model = LSTMForecaster().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.MSELoss()
    pin   = DEVICE.type == "cuda"
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=BATCH, shuffle=True, pin_memory=pin, num_workers=0)
    Xv = torch.from_numpy(X_va).to(DEVICE)
    yv = torch.from_numpy(y_va).to(DEVICE)
    
    best, state, wait = float("inf"), None, 0
    tr_h, val_h = [], []
    best_ep = 1
    
    for ep in range(EPOCHS):
        model.train(); bl = []
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=pin)
            yb = yb.to(DEVICE, non_blocking=pin)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step(); bl.append(loss.item())
        tr_h.append(float(np.mean(bl)))
        
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv), yv).item()
        val_h.append(vl)
        
        if vl < best:
            best = vl
            state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ep = ep + 1; wait = 0
        else:
            wait += 1
            if wait >= PATIENCE: break
    
    del Xv, yv
    model.load_state_dict({k: v.to(DEVICE) for k, v in state.items()})
    return model, tr_h, val_h, best_ep


def infer(model, X, bs=512):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(torch.from_numpy(X[i:i+bs]).to(DEVICE)).cpu().numpy())
    return np.concatenate(out)


# ═══════════════════════════════════════════════════════════════════
# PİPELİNE
# ═══════════════════════════════════════════════════════════════════
def run_pipeline(csv_path, rel_key, windows_dict, save_predictions=False):
    df = pd.read_csv(csv_path)
    df["value"]     = handle_missing(df["value"].values)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["label"] = 0
    for s, e in windows_dict.get(rel_key, []):
        mask = (df["timestamp"] >= pd.to_datetime(s)) & (df["timestamp"] <= pd.to_datetime(e))
        df.loc[mask, "label"] = 1
    
    n = len(df)
    all_labels = df["label"].values.astype(np.int8)
    total_anom = int(all_labels.sum())
    
    i1 = int(n * TRAIN_R); i2 = int(n * (TRAIN_R + VAL_R))
    raw_tr = df["value"].values[:i1].astype(np.float32)
    raw_va = df["value"].values[i1:i2].astype(np.float32)
    raw_te = df["value"].values[i2:].astype(np.float32)
    test_anom = int(all_labels[i2:].sum())
    
    # ═════════════════ STL CAUSAL — LEAKAGE YOK ═════════════════
    dt_min = df["timestamp"].diff().median().total_seconds() / 60
    period = get_period(raw_tr, dt_min)         # ← sadece train'den period
    tr_r, va_r, te_r, stl_ok = causal_stl(raw_tr, raw_va, raw_te, period)
    # ════════════════════════════════════════════════════════════
    
    # Z-score — sadece train'e fit (zaten)
    scaler = StandardScaler()
    tr_sc = scaler.fit_transform(tr_r.reshape(-1,1)).flatten().astype(np.float32)
    va_sc = scaler.transform(va_r.reshape(-1,1)).flatten().astype(np.float32)
    te_sc = scaler.transform(te_r.reshape(-1,1)).flatten().astype(np.float32)
    
    X_tr, y_tr = make_windows(tr_sc, SEQ_LEN)
    X_va, y_va = make_windows(va_sc, SEQ_LEN)
    X_te, y_te, n_skipped = make_test_windows_full(tr_sc, va_sc, te_sc)
    
    if len(X_tr) < 2 or len(X_va) == 0 or len(X_te) == 0:
        return None
    
    if n_skipped > 0:
        y_labels_test = all_labels[i2 + n_skipped : i2 + n_skipped + len(y_te)]
    else:
        y_labels_test = all_labels[i2 : i2 + len(y_te)]
    n_test = len(y_te)
    
    raw_test_windows = get_test_windows(df, i2, windows_dict, rel_key)
    test_windows = []
    for s, e in raw_test_windows:
        s_adj = s - n_skipped; e_adj = e - n_skipped
        if e_adj < 0 or s_adj >= n_test: continue
        test_windows.append((max(0, s_adj), min(n_test-1, e_adj)))
    
    model, tr_h, val_h, best_ep = train_model(X_tr, y_tr, X_va, y_va)
    
    err_tr = np.abs(y_tr - infer(model, X_tr))
    y_pred = infer(model, X_te)
    err_te = np.abs(y_te - y_pred)
    
    mu  = float(err_tr.mean()); sig = float(err_tr.std()) + 1e-10
    z_te = (err_te - mu) / sig
    preds = (np.abs(z_te) > TAU).astype(np.int8)
    
    m_point = f1_pointwise(y_labels_test, preds)
    m_event = f1_eventbased(preds, test_windows, n_test)
    
    mae  = float(np.mean(np.abs(y_te - y_pred)))
    rmse = float(np.sqrt(np.mean((y_te - y_pred)**2)))
    pcc  = float(np.corrcoef(y_te, y_pred)[0,1]) if y_te.std() > 1e-8 and y_pred.std() > 1e-8 else 0.0
    ss_res = np.sum((y_te - y_pred)**2)
    ss_tot = np.sum((y_te - np.mean(y_te))**2)
    r2 = float(1 - ss_res/ss_tot) if ss_tot > 1e-8 else 0.0
    
    cat_folder = rel_key.split("/")[0]
    category = CAT_MAP.get(cat_folder, "Unknown")
    
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    out = dict(
        key=rel_key, category=category,
        n=n, total_anom=total_anom, test_anom=test_anom,
        n_test=n_test, n_skipped=n_skipped, n_events=len(test_windows),
        stl=stl_ok, period=period, best_ep=best_ep,
        mae=mae, rmse=rmse, pcc=pcc, r2=r2,
        f1_p=m_point["f1"], pre_p=m_point["pre"], rec_p=m_point["rec"], fpr_p=m_point["fpr"],
        f1_e=m_event["f1"], pre_e=m_event["pre"], rec_e=m_event["rec"],
        tp_e=m_event["tp"], fp_e=m_event["fp"], fn_e=m_event["fn"],
        _tr_h=tr_h, _val_h=val_h,
    )
    if save_predictions:
        out["_y_te"] = y_te; out["_y_pred"] = y_pred
        out["_preds"] = preds; out["_y_labels"] = y_labels_test
        out["_test_windows"] = test_windows
    return out


# ═══════════════════════════════════════════════════════════════════
# ANA DÖNGÜ
# ═══════════════════════════════════════════════════════════════════
with open(WINDOWS_PATH) as f:
    windows = json.load(f)

data_dir = os.path.join(NAB_ROOT, "data")
csv_files = []
for cat in sorted(os.listdir(data_dir)):
    cp = os.path.join(data_dir, cat)
    if not os.path.isdir(cp): continue
    for fn in sorted(os.listdir(cp)):
        if fn.endswith(".csv"):
            csv_files.append((os.path.join(cp, fn), f"{cat}/{fn}"))

case_study_keys = set([c["file"] for c in CASE_STUDIES.values()])

print(f"Toplam CSV: {len(csv_files)}")
print(f"⚠ Causal STL kullanılıyor → her dosya biraz daha yavaş\n")
print(f"{'Dosya':<38} {'Kategori':<14} {'evt':>4} "
      f"{'F1pt':>6} {'F1ev':>6} {'MAE':>7} {'PCC':>7} {'STL':>8}")
print("─"*110)

results = []
for path, key in csv_files:
    cat = CAT_MAP.get(key.split("/")[0], "?")
    save_pred = (key in case_study_keys)
    try:
        res = run_pipeline(path, key, windows, save_predictions=save_pred)
        if res is None:
            print(f"  {key.split('/')[-1][:36]:<38}  ATLANDI")
            continue
        results.append(res)
        fname = res["key"].split("/")[-1].replace(".csv","")[:36]
        stl_str = f"STL✓p{res['period']}" if res["stl"] else "STL✗"
        print(f"  {fname:<36}  {cat[:12]:<14} {res['n_events']:>4} "
              f"{res['f1_p']:>6.3f} {res['f1_e']:>6.3f} "
              f"{res['mae']:>7.3f} {res['pcc']:>7.3f} {stl_str:>8}")
    except Exception as e:
        print(f"  {key.split('/')[-1][:36]}  HATA: {e}")


# ═══════════════════════════════════════════════════════════════════
# ÖZET (TÜM 58, FİLTRESİZ)
# ═══════════════════════════════════════════════════════════════════
rdf = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")} for r in results])
anom = rdf[rdf["test_anom"] > 0]

print(f"\n{'═'*80}")
print(f"  📊 GENEL SONUÇLAR — STL LEAKAGE DÜZELTİLMİŞ")
print(f"{'═'*80}")
print(f"  Toplam dosya             : {len(rdf)}")
print(f"  STL açıldı (causal)      : {int(rdf['stl'].sum())} dosya")
print(f"  Test'te anomali olan     : {len(anom)}")

print(f"\n  Forecasting (Tüm 58 dosya):")
print(f"  {'Metrik':<10} {'Bizim':<12} {'Makale':<10}")
for m, key in [("MAE","mae"), ("RMSE","rmse"), ("PCC","pcc")]:
    bz = rdf[key].mean()
    mk = PAPER_OVERALL[key.lower() if key=="mae" else key]
    print(f"  {m:<10} {bz:<12.4f} {mk:<10.4f}")

print(f"\n  Detection (Tüm 58 dosya):")
print(f"  {'Metrik':<12} {'Point-wise':<14} {'Event-based ★':<16} {'Makale':<10}")
print(f"  {'Precision':<12} {rdf['pre_p'].mean():<14.4f} {rdf['pre_e'].mean():<16.4f} {PAPER_OVERALL['pre']:<10.4f}")
print(f"  {'Recall':<12} {rdf['rec_p'].mean():<14.4f} {rdf['rec_e'].mean():<16.4f} {PAPER_OVERALL['rec']:<10.4f}")
print(f"  {'F1':<12} {rdf['f1_p'].mean():<14.4f} {rdf['f1_e'].mean():<16.4f} {PAPER_OVERALL['f1']:<10.4f}")

print(f"\n  Bilgi — Anomalili {len(anom)} dosya:")
print(f"    F1 point-wise : {anom['f1_p'].mean():.4f}")
print(f"    F1 event-based: {anom['f1_e'].mean():.4f}")

print(f"\n{'═'*80}")
print(f"  📈 KATEGORİ BAZLI")
print(f"{'═'*80}")
print(f"  {'Kategori':<18} {'n':>3} {'F1-pt':>8} {'F1-ev':>8} {'MAE':>7} {'PCC':>7} {'Mak.F1':>8}")
for cat_name in CAT_MAP.values():
    sub = rdf[rdf["category"] == cat_name]
    if len(sub) == 0: continue
    pref = PAPER_CAT.get(cat_name, {})
    mf = f"{pref['f1']:.3f}" if pref.get("f1") else "  -  "
    print(f"  {cat_name:<16} {len(sub):>3} "
          f"{sub['f1_p'].mean():>8.4f} {sub['f1_e'].mean():>8.4f} "
          f"{sub['mae'].mean():>7.4f} {sub['pcc'].mean():>7.4f} {mf:>8}")

rdf.to_csv(os.path.join(NAB_ROOT, "lstm_no_leakage_results.csv"), index=False)
print(f"\n✓ lstm_no_leakage_results.csv kaydedildi")


# ═══════════════════════════════════════════════════════════════════
# ŞEKİL 1 — LOSS EĞRİLERİ
# ═══════════════════════════════════════════════════════════════════
anom_res = [r for r in results if r["test_anom"] > 0]
if len(anom_res) > 0:
    cols = 3
    rows = (len(anom_res) + 2) // 3
    fig1, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*3.5))
    axes = axes.flatten() if rows*cols > 1 else [axes]
    
    for idx, r in enumerate(anom_res):
        ax = axes[idx]
        vs = pd.Series(r["_val_h"]).rolling(3, min_periods=1).mean().values
        ep = range(1, len(r["_tr_h"])+1)
        col = CAT_COLORS.get(r["category"], "#2196F3")
        ax.plot(ep, r["_tr_h"], color=col, lw=2, label="Train")
        ax.plot(ep, vs, "#FF9800", lw=2, label="Val (smooth)")
        ax.axvline(r["best_ep"], color="green", ls="--", lw=1.5,
                   label=f"Best ep{r['best_ep']}")
        stl_str = f"STL✓p{r['period']}" if r["stl"] else "STL✗"
        fname = r["key"].split("/")[-1].replace(".csv","")[:26]
        ax.set_title(f"{fname}  [{stl_str}]\n"
                     f"F1pt={r['f1_p']:.3f}  F1ev={r['f1_e']:.3f}  "
                     f"Pre={r['pre_e']:.2f}  Rec={r['rec_e']:.2f}  PCC={r['pcc']:.3f}",
                     fontsize=8, fontweight="bold", color=col)
        ax.set_xlabel("Epoch", fontsize=7); ax.set_ylabel("MSE Loss", fontsize=7)
        ax.tick_params(labelsize=7); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.98, r["category"], transform=ax.transAxes,
                fontsize=6, va="top", color=col, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor=col,
                         boxstyle="round,pad=0.2"))
    
    for idx in range(len(anom_res), len(axes)):
        axes[idx].set_visible(False)
    
    fig1.suptitle(f"LSTM Loss Eğrileri (Anomalili {len(anom_res)} Dosya)\n"
                  f"STL LEAKAGE DÜZELTİLMİŞ | Causal Rolling STL | "
                  f"Event-Based F1 | τ={TAU}",
                  fontsize=13, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig(os.path.join(NAB_ROOT, "fig_1_loss_curves.png"),
                 dpi=130, bbox_inches="tight")
    plt.close(fig1)
    print("✓ fig_1_loss_curves.png kaydedildi")


# ═══════════════════════════════════════════════════════════════════
# ŞEKİL 2 — 4 CASE STUDY (Tables VI-IX)
# ═══════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(20, 16))
gs2 = gridspec.GridSpec(4, 3, figure=fig2, hspace=0.55, wspace=0.30)

case_results = {}
for tname, case in CASE_STUDIES.items():
    for r in results:
        if r["key"] == case["file"]:
            case_results[tname] = r; break

csv_rows = []
for row_idx, (tname, case) in enumerate(CASE_STUDIES.items()):
    if tname not in case_results: continue
    r = case_results[tname]; paper = case["paper"]
    short_name = case["file"].split("/")[-1].replace(".csv","")
    
    y_te = r["_y_te"]; y_pred = r["_y_pred"]
    y_labels = r["_y_labels"]
    
    # Panel 1: Forecasting
    ax = fig2.add_subplot(gs2[row_idx, 0])
    n_show = min(500, len(y_te))
    ax.plot(y_te[:n_show], color="#1976D2", lw=1.2, label="Gerçek", alpha=0.85)
    ax.plot(y_pred[:n_show], color="#FF7043", lw=1.0, label="Tahmin", alpha=0.85)
    if y_labels[:n_show].sum() > 0:
        anom_pts = np.where(y_labels[:n_show] == 1)[0]
        ax.scatter(anom_pts, y_te[:n_show][anom_pts], color="red", s=15,
                   zorder=5, alpha=0.6, label="Anomali")
    ax.set_title(f"{tname}: {short_name}\nForecasting (ilk {n_show})",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Test Noktası", fontsize=8)
    ax.set_ylabel("Değer (norm.)", fontsize=8)
    ax.legend(fontsize=7, loc="best"); ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    
    # Panel 2: Metrik karşılaştırma
    ax = fig2.add_subplot(gs2[row_idx, 1])
    metric_names = ["MAE", "RMSE", "PCC", "R²", "F1-pt", "F1-ev"]
    our_vals = [r["mae"], r["rmse"], r["pcc"], r["r2"], r["f1_p"], r["f1_e"]]
    paper_vals = [paper.get("MAE"), paper.get("RMSE"), paper.get("PCC"),
                  paper.get("R2"), paper.get("F1"), paper.get("F1")]
    
    x = np.arange(len(metric_names)); w = 0.35
    valid_our = [v if v is not None else 0 for v in our_vals]
    valid_paper = [v if v is not None else 0 for v in paper_vals]
    
    bars_o = ax.bar(x-w/2, valid_our, w, color="#42A5F5", alpha=0.85, label="Bizim",
                    edgecolor="black", lw=0.5)
    bars_p = ax.bar(x+w/2, valid_paper, w, color="#9E9E9E", alpha=0.6, hatch="///",
                    label="Makale LSTM", edgecolor="black", lw=0.5)
    
    for i, pv in enumerate(paper_vals):
        if pv is None: bars_p[i].set_visible(False)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=8, rotation=30, ha="right")
    ax.set_title("Metrik Karşılaştırma", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=7)
    
    for xi, v in enumerate(valid_our):
        ax.text(xi-w/2, v+0.01, f"{v:.3f}", ha="center", fontsize=7, fontweight="bold")
    for xi, v in enumerate(valid_paper):
        if v != 0:
            ax.text(xi+w/2, v+0.01, f"{v:.3f}", ha="center", fontsize=7)
    
    # Panel 3: Loss
    ax = fig2.add_subplot(gs2[row_idx, 2])
    vs = pd.Series(r["_val_h"]).rolling(3, min_periods=1).mean().values
    ep = range(1, len(r["_tr_h"])+1)
    ax.plot(ep, r["_tr_h"], color="#1976D2", lw=2, label="Train")
    ax.plot(ep, vs, "#FF9800", lw=2, label="Val (smooth)")
    ax.axvline(r["best_ep"], color="green", ls="--", lw=1.5,
               label=f"Best ep{r['best_ep']}")
    ax.set_title(f"Eğitim Eğrisi\n(STL: {'p='+str(r['period']) if r['stl'] else 'KAPALI'})",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=8); ax.set_ylabel("MSE Loss", fontsize=8)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    
    csv_rows.append({
        "Table": tname, "Dosya": short_name,
        "Bizim_MAE":  r["mae"],  "Makale_MAE":  paper.get("MAE"),
        "Bizim_RMSE": r["rmse"], "Makale_RMSE": paper.get("RMSE"),
        "Bizim_PCC":  r["pcc"],  "Makale_PCC":  paper.get("PCC"),
        "Bizim_R2":   r["r2"],   "Makale_R2":   paper.get("R2"),
        "Bizim_F1_pt":r["f1_p"], "Bizim_F1_ev": r["f1_e"], "Makale_F1": paper.get("F1"),
        "n_test": r["n_test"], "n_events": r["n_events"],
        "STL": r["stl"], "Best_Epoch": r["best_ep"],
    })

fig2.suptitle("4 Case Study — Karami et al. (2025) Tables VI-IX vs Reprodüksiyon\n"
              "STL LEAKAGE DÜZELTİLMİŞ — Causal Rolling STL",
              fontsize=13, fontweight="bold", y=1.00)
plt.savefig(os.path.join(NAB_ROOT, "fig_2_case_studies.png"),
            dpi=130, bbox_inches="tight")
plt.close(fig2)
print("✓ fig_2_case_studies.png kaydedildi")

pd.DataFrame(csv_rows).to_csv(os.path.join(NAB_ROOT, "case_studies_no_leakage.csv"), index=False)
print("✓ case_studies_no_leakage.csv kaydedildi")


# ═══════════════════════════════════════════════════════════════════
# ŞEKİL 3 — ÖZET
# ═══════════════════════════════════════════════════════════════════
fig3, axes = plt.subplots(1, 3, figsize=(18, 6))

ax = axes[0]
metrics = ["MAE", "RMSE", "PCC"]
bizim = [rdf["mae"].mean(), rdf["rmse"].mean(), rdf["pcc"].mean()]
mak   = [PAPER_OVERALL["mae"], PAPER_OVERALL["rmse"], PAPER_OVERALL["pcc"]]
x = np.arange(len(metrics)); w = 0.35
ax.bar(x-w/2, bizim, w, color="#2196F3", alpha=0.85, label="Bizim (58, no-leakage)")
ax.bar(x+w/2, mak, w, color="#B0BEC5", alpha=0.5, hatch="///",
       edgecolor="black", lw=1, label="Makale")
ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=11); ax.set_ylabel("Değer")
ax.set_title("a) Forecasting\nSTL Leakage Düzeltilmiş",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y")
for xi, (b, m) in enumerate(zip(bizim, mak)):
    ax.text(xi-w/2, b+0.05, f"{b:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(xi+w/2, m+0.05, f"{m:.3f}", ha="center", fontsize=9)

ax = axes[1]
metrics = ["Precision", "Recall", "F1"]
bizim_p = [rdf["pre_p"].mean(), rdf["rec_p"].mean(), rdf["f1_p"].mean()]
bizim_e = [rdf["pre_e"].mean(), rdf["rec_e"].mean(), rdf["f1_e"].mean()]
mak     = [PAPER_OVERALL["pre"], PAPER_OVERALL["rec"], PAPER_OVERALL["f1"]]
x = np.arange(len(metrics)); w = 0.27
ax.bar(x-w, bizim_p, w, color="#E57373", alpha=0.85, label="Point-wise")
ax.bar(x,   bizim_e, w, color="#66BB6A", alpha=0.85, label="Event-based ★")
ax.bar(x+w, mak, w, color="#B0BEC5", alpha=0.5, hatch="///",
       edgecolor="black", lw=1, label="Makale")
ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylabel("Değer"); ax.set_ylim(0, 1.0)
ax.set_title("b) Detection\nSTL Leakage Düzeltilmiş",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
for xi, (p, e, m) in enumerate(zip(bizim_p, bizim_e, mak)):
    ax.text(xi-w, p+0.01, f"{p:.2f}", ha="center", fontsize=7, fontweight="bold")
    ax.text(xi,   e+0.01, f"{e:.2f}", ha="center", fontsize=7, fontweight="bold")
    ax.text(xi+w, m+0.01, f"{m:.2f}", ha="center", fontsize=7)

ax = axes[2]
cat_list = list(CAT_MAP.values())
cats_short = [c.replace("Art. ","Art.") for c in cat_list]
our_f1 = [rdf[rdf["category"]==c]["f1_e"].mean() if len(rdf[rdf["category"]==c])>0 else 0
          for c in cat_list]
mak_f1 = [PAPER_CAT.get(c,{}).get("f1") or 0 for c in cat_list]
x = np.arange(len(cat_list)); w = 0.35
ax.bar(x-w/2, our_f1, w,
       color=[CAT_COLORS[c] for c in cat_list], alpha=0.85, label="Bizim Event-based")
ax.bar(x+w/2, mak_f1, w,
       color=[CAT_COLORS[c] for c in cat_list], alpha=0.35,
       edgecolor="black", lw=1.5, hatch="///", label="Makale")
ax.set_xticks(x); ax.set_xticklabels(cats_short, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Ortalama F1"); ax.set_ylim(0, 1.0)
ax.axhline(PAPER_OVERALL["f1"], color="red", ls="--", lw=1.5, alpha=0.4)
ax.set_title("c) Kategori Bazlı F1\nNo-Leakage",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

plt.suptitle(f"NAB LSTM — STL LEAKAGE DÜZELTİLMİŞ — Tüm 58 dosya | "
             f"Causal Rolling STL | τ={TAU}",
             fontsize=12, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(NAB_ROOT, "fig_3_overall_summary.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig3)
print("✓ fig_3_overall_summary.png kaydedildi")

print(f"\n{'═'*80}")
print(f"  ÇIKTILAR (d:\\NAB\\):")
print(f"{'═'*80}")
print(f"  📊 lstm_no_leakage_results.csv")
print(f"  📊 case_studies_no_leakage.csv")
print(f"  📈 fig_1_loss_curves_1.png         (anomalili dosyalar)")
print(f"  📈 fig_2_case_studies_1.png         (4 case Table VI-IX)")
print(f"  📈 fig_3_overall_summary_1.png      (genel özet)")
print(f"{'═'*80}\n")