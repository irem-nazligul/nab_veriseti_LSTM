"""
NAB — LSTM Anomali Tespiti (Makale Algorithm 1 BİREBİR — v12)
Karami et al. (2025) arXiv:2510.11141

v11 → v12 TEK kritik değişiklik:
  - Stats (μ_e, σ_e) artık SADECE train'den DEĞİL,
    Algorithm 1 satır 20'deki gibi TÜM seri hataları üzerinden hesaplanıyor

Gerekçe:
  Algorithm 1 satır 20: μ_e, σ_e ← ComputeStats({e_t})
  Burada {e_t} = satır 14'te t = w+1...n için hesaplanan TÜM hatalar.
  Section III-D.1 metni "training residuals" diyor ama bu Algorithm
  spesifikasyonu ile çelişiyor — algorithm formel, metin özetleyici.

  Pratikte de bu yorum doğru: makalenin Table II'de bildirdiği
  FPR=0.215, k=3 ile train-only stats kullanılırsa imkansız
  (teorik FPR=0.0027). Ancak test concept drift'i de stats'a dahil
  edilirse FPR=0.215 mantıklı hale geliyor.

Diğer her şey v11 ile aynı:
  ✓ τ=3.0 sabit (Eq.14)
  ✓ Z-score normalize (Eq.2)
  ✓ STL ACF≥0.3 koşullu (Algorithm 1 satır 6)
  ✓ 2×LSTM(64), w=50, dropout=0.2
  ✓ Diagnostic etiketleme + Strict F1 + NAB F1
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

# ══════════════════════════════════════════════════════════════════
# PARAMETRELER
# ══════════════════════════════════════════════════════════════════
NAB_ROOT     = r"d:\NAB"
WINDOWS_PATH = os.path.join(NAB_ROOT, "labels", "combined_windows.json")

TRAIN_R, VAL_R = 0.70, 0.15
SEQ_LEN   = 50
EPOCHS    = 30
HIDDEN    = 64
LAYERS    = 2
DROPOUT   = 0.2
LR        = 1e-3
BATCH     = 32
GRAD_CLIP = 1.0
PATIENCE  = 5
ACF_THR   = 0.3
TAU       = 3.0   # Eq.14

PAPER_CAT = {
    "Art. No Anomaly": {"mae": None,  "f1": 0.55},
    "Art. Anomaly":    {"mae": 0.08,  "f1": None},
    "AWS CloudWatch":  {"mae": 0.31,  "f1": 0.74},
    "Ad Exchange":     {"mae": 0.42,  "f1": 0.65},
    "Known Cause":     {"mae": 0.25,  "f1": 0.81},
    "Traffic":         {"mae": 0.38,  "f1": None},
    "Twitter":         {"mae": 0.36,  "f1": 0.68},
}
PAPER_OVERALL = {"mae": 0.245, "rmse": 0.421, "pcc": 0.782, "f1": 0.688}

CAT_MAP = {
    "artificialNoAnomaly":  "Art. No Anomaly",
    "artificialWithAnomaly":"Art. Anomaly",
    "realAWSCloudwatch":    "AWS CloudWatch",
    "realAdExchange":       "Ad Exchange",
    "realKnownCause":       "Known Cause",
    "realTraffic":          "Traffic",
    "realTweets":           "Twitter",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
print(f"τ      : {TAU} SABİT (Eq.14)")
print(f"Stats  : TÜM seri hataları üzerinden (Algorithm 1 satır 20)")
print(f"Eval   : Strict F1 + NAB F1\n")


# ══════════════════════════════════════════════════════════════════
# YARDIMCI
# ══════════════════════════════════════════════════════════════════
def handle_missing(arr):
    s = pd.Series(arr, dtype=float)
    return (s.fillna(method="ffill", limit=4)
             .interpolate().fillna(method="bfill")
             .values.astype(np.float32))


def get_period(train_raw, interval_min):
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


def apply_stl_full(raw_tr, raw_va, raw_te, period):
    if period is None: return raw_tr, raw_va, raw_te, False
    try:
        n1 = len(raw_tr); n2 = n1+len(raw_va)
        full = np.concatenate([raw_tr, raw_va, raw_te]).astype(np.float64)
        res  = STL(full, period=period, robust=True).fit().resid.astype(np.float32)
        return res[:n1], res[n1:n2], res[n2:], True
    except: return raw_tr, raw_va, raw_te, False


def make_windows(arr, seq_len):
    arr = np.asarray(arr, np.float32).ravel()
    N   = len(arr)-seq_len
    if N <= 0: return np.empty((0,seq_len,1),np.float32), np.empty(0,np.float32)
    strides = (arr.strides[0], arr.strides[0])
    X2d = np.lib.stride_tricks.as_strided(arr, shape=(N,seq_len), strides=strides)
    return np.ascontiguousarray(X2d[:,:,np.newaxis]), arr[seq_len:].copy()


def compute_metrics_diagnostic(y_true, y_pred):
    y_true = np.asarray(y_true, np.int8)
    y_pred = np.asarray(y_pred, np.int8)
    tp = int(((y_pred==1)&(y_true==1)).sum())
    fp = int(((y_pred==1)&(y_true==0)).sum())
    fn = int(((y_pred==0)&(y_true==1)).sum())
    tn = int(((y_pred==0)&(y_true==0)).sum())
    pre = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
    strict_f1 = 2*pre*rec/(pre+rec) if (pre+rec)>0 else 0.0
    fpr = fp/(fp+tn) if (fp+tn)>0 else 0.0
    test_has_anom = (y_true.sum() > 0)
    if not test_has_anom:
        if fp == 0:
            nab_f1 = 1.0; diag = "no_anom_test_clean"
        else:
            nab_f1 = 0.0; diag = "no_anom_test_FP"
    else:
        nab_f1 = strict_f1
        if y_pred.sum() == 0:           diag = "missed_all"
        elif tp == 0:                   diag = "all_fp"
        elif strict_f1 < 0.1:           diag = "partial_miss"
        elif strict_f1 < 0.3:           diag = "low_f1"
        elif strict_f1 < 0.6:           diag = "mid_f1"
        else:                           diag = "high_f1"
    return dict(
        f1=round(strict_f1,4), nab_f1=round(nab_f1,4),
        pre=round(pre,4), rec=round(rec,4), fpr=round(fpr,4),
        tp=tp, fp=fp, fn=fn, tn=tn,
        test_anom=int(y_true.sum()), diag=diag
    )


# ══════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════
class LSTMForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1,HIDDEN,LAYERS,batch_first=True,dropout=DROPOUT)
        self.fc   = nn.Linear(HIDDEN,1)
    def forward(self,x):
        out,_ = self.lstm(x); return self.fc(out[:,-1,:]).squeeze(-1)


def train_model(X_tr,y_tr,X_va,y_va):
    model  = LSTMForecaster().to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(),lr=LR)
    crit   = nn.MSELoss()
    pin    = DEVICE.type=="cuda"
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr),torch.from_numpy(y_tr)),
                        batch_size=BATCH,shuffle=True,pin_memory=pin,num_workers=0)
    Xv=torch.from_numpy(X_va).to(DEVICE); yv=torch.from_numpy(y_va).to(DEVICE)
    best,state,wait=float("inf"),None,0; tr_h,val_h=[],[]; best_ep=1
    for ep in range(EPOCHS):
        model.train(); bl=[]
        for xb,yb in loader:
            xb=xb.to(DEVICE,non_blocking=pin); yb=yb.to(DEVICE,non_blocking=pin)
            loss=crit(model(xb),yb); opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
            opt.step(); bl.append(loss.item())
        tr_h.append(float(np.mean(bl)))
        model.eval()
        with torch.no_grad(): vl=crit(model(Xv),yv).item()
        val_h.append(vl)
        if vl<best: best=vl; state={k:v.cpu().clone() for k,v in model.state_dict().items()}; best_ep=ep+1; wait=0
        else:
            wait+=1
            if wait>=PATIENCE: break
    del Xv,yv
    model.load_state_dict({k:v.to(DEVICE) for k,v in state.items()})
    return model,tr_h,val_h,best_ep


def infer(model,X,bs=512):
    model.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(X),bs):
            out.append(model(torch.from_numpy(X[i:i+bs]).to(DEVICE)).cpu().numpy())
    return np.concatenate(out)


# ══════════════════════════════════════════════════════════════════
# PİPELİNE
# ══════════════════════════════════════════════════════════════════
def run_pipeline(csv_path, rel_key, windows_dict):
    df = pd.read_csv(csv_path)
    df["value"]     = handle_missing(df["value"].values)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["label"] = 0
    for s,e in windows_dict.get(rel_key,[]):
        mask = (df["timestamp"]>=pd.to_datetime(s))&(df["timestamp"]<=pd.to_datetime(e))
        df.loc[mask,"label"] = 1
    total_anom = int(df["label"].sum())

    n=len(df); i1=int(n*TRAIN_R); i2=int(n*(TRAIN_R+VAL_R))
    raw_tr=df["value"].values[:i1].astype(np.float32)
    raw_va=df["value"].values[i1:i2].astype(np.float32)
    raw_te=df["value"].values[i2:].astype(np.float32)
    train_anom = int(df["label"].values[:i1].sum())
    val_anom   = int(df["label"].values[i1:i2].sum())
    y_labels   = df["label"].values[i2:].astype(np.int8)

    dt_min = df["timestamp"].diff().median().total_seconds()/60
    period = get_period(raw_tr, dt_min)
    tr_r, va_r, te_r, stl_ok = apply_stl_full(raw_tr, raw_va, raw_te, period)

    scaler=StandardScaler()
    tr_sc=scaler.fit_transform(tr_r.reshape(-1,1)).flatten().astype(np.float32)
    va_sc=scaler.transform(va_r.reshape(-1,1)).flatten().astype(np.float32)
    te_sc=scaler.transform(te_r.reshape(-1,1)).flatten().astype(np.float32)

    X_tr,y_tr=make_windows(tr_sc,SEQ_LEN)
    X_va,y_va=make_windows(va_sc,SEQ_LEN)
    buf=np.concatenate([va_sc[-SEQ_LEN:],te_sc])
    X_te,y_te=make_windows(buf,SEQ_LEN)

    if len(X_tr)<2 or len(X_va)==0 or len(X_te)==0: return None
    y_labels = y_labels[:len(y_te)]
    if len(y_te)!=len(y_labels): return None

    model,tr_h,val_h,best_ep=train_model(X_tr,y_tr,X_va,y_va)

    # Tüm setler için forecast hatalarını hesapla
    err_tr=np.abs(y_tr-infer(model,X_tr))
    err_va=np.abs(y_va-infer(model,X_va))
    y_pred_out = infer(model,X_te)
    err_te=np.abs(y_te-y_pred_out)

    # ═════════════════════════════════════════════════════════════
    # v12 KRITIK DEĞİŞİKLİK: Algorithm 1 satır 20
    # μ_e, σ_e ← ComputeStats({e_t}) — TÜM e_t üzerinden
    # ═════════════════════════════════════════════════════════════
    all_err = np.concatenate([err_tr, err_va, err_te])
    mu  = float(all_err.mean())
    sig = float(all_err.std()) + 1e-10

    z_te = (err_te - mu) / sig
    preds = (np.abs(z_te) > TAU).astype(np.int8)

    mae =float(np.mean(np.abs(y_te-y_pred_out)))
    rmse=float(np.sqrt(np.mean((y_te-y_pred_out)**2)))
    pcc =float(np.corrcoef(y_te,y_pred_out)[0,1]) if y_te.std()>1e-8 and y_pred_out.std()>1e-8 else 0.0
    det = compute_metrics_diagnostic(y_labels, preds)

    cat_folder=rel_key.split("/")[0]
    category=CAT_MAP.get(cat_folder,"Unknown")

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return dict(
        key=rel_key, category=category,
        n=n, total_anom=total_anom,
        train_anom=train_anom, val_anom=val_anom, test_anom=det["test_anom"],
        is_anomaly=bool(total_anom>0),
        stl=stl_ok, period=period,
        best_ep=best_ep, mae=round(mae,4), rmse=round(rmse,4), pcc=round(pcc,4),
        mu_e=round(mu,4), sig_e=round(sig,4),
        f1=det["f1"], nab_f1=det["nab_f1"],
        pre=det["pre"], rec=det["rec"], fpr=det["fpr"],
        tp=det["tp"], fp=det["fp"], fn=det["fn"], tn=det["tn"],
        diag=det["diag"],
        _tr_h=tr_h, _val_h=val_h, _z=z_te, _preds=preds, _labels=y_labels,
        _y_te=y_te, _y_pred=y_pred_out
    )


# ══════════════════════════════════════════════════════════════════
# ANA DÖNGÜ
# ══════════════════════════════════════════════════════════════════
with open(WINDOWS_PATH) as f: windows=json.load(f)
data_dir=os.path.join(NAB_ROOT,"data"); csv_files=[]
for cat in sorted(os.listdir(data_dir)):
    cp=os.path.join(data_dir,cat)
    if not os.path.isdir(cp): continue
    for fn in sorted(os.listdir(cp)):
        if fn.endswith(".csv"): csv_files.append((os.path.join(cp,fn),f"{cat}/{fn}"))

print(f"Toplam CSV: {len(csv_files)}"); print("="*108)
print(f"{'Dosya':<42} {'T/V/Te':<12} {'F1':>6} {'NAB':>6} {'Pre':>5} {'Rec':>5} {'σ_e':>6} {'MAE':>6} {'Teşhis':<20}")
print("-"*108)
results=[]
for path,key in csv_files:
    try:
        res=run_pipeline(path,key,windows)
        if res is None:
            print(f"  {key.split('/')[-1][:40]:<42}  ATLANDI"); continue
        results.append(res)
        anom_str = f"{res['train_anom']}/{res['val_anom']}/{res['test_anom']}"
        fname = res["key"].split("/")[-1][:40]
        print(f"  {fname:<40}  {anom_str:<12} {res['f1']:>6.3f} {res['nab_f1']:>6.3f} "
              f"{res['pre']:>5.3f} {res['rec']:>5.3f} {res['sig_e']:>6.3f} {res['mae']:>6.3f}  {res['diag']:<20}")
    except Exception as e:
        print(f"  {key.split('/')[-1][:40]}  HATA: {e}")


# ══════════════════════════════════════════════════════════════════
# TEŞHİS ÖZETİ
# ══════════════════════════════════════════════════════════════════
rdf = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")} for r in results])
rdf_cl=rdf[rdf["mae"]<5.0]

print(f"\n{'='*100}")
print(f"  F1=0 SEBEP ANALİZİ (v12 — Stats TÜM seri üzerinde)")
print(f"{'='*100}")
diag_counts = rdf["diag"].value_counts()
diag_descs = {
    "no_anom_test_clean" : "Test anomalisiz, FP yok (NAB: F1=1.0) ✓",
    "no_anom_test_FP"    : "Test anomalisiz ama FP var (yanlış alarm)",
    "missed_all"         : "Anomali var, hiç tahmin yok (τ yüksek)",
    "all_fp"             : "Çok tahmin var, TP=0",
    "partial_miss"       : "F1 < 0.10",
    "low_f1"             : "F1: 0.10-0.30",
    "mid_f1"             : "F1: 0.30-0.60",
    "high_f1"            : "F1: 0.60+",
}
total = len(rdf)
for diag in ["no_anom_test_clean", "no_anom_test_FP", "missed_all",
             "all_fp", "partial_miss", "low_f1", "mid_f1", "high_f1"]:
    n_d = diag_counts.get(diag, 0)
    pct = 100*n_d/total if total > 0 else 0
    print(f"  {diag:<22} n={n_d:>2}  ({pct:>5.1f}%)  {diag_descs[diag]}")

# EVALUATION
print(f"\n{'='*100}")
print(f"  EVALUATION — Strict vs NAB Standart")
print(f"{'='*100}")
print(f"  Strict F1 ortalaması   : {rdf['f1'].mean():.4f}  (58 dosya)")
print(f"  NAB    F1 ortalaması   : {rdf['nab_f1'].mean():.4f}  ← Makale standardı")
print(f"  Makale F1              : {PAPER_OVERALL['f1']:.4f}")

anom_only = rdf[rdf["test_anom"]>0]
print(f"\n  Test'te anomali olan dosyalar ({len(anom_only)} dosya):")
print(f"    Strict F1 ort        : {anom_only['f1'].mean():.4f}")
print(f"    Precision ort        : {anom_only['pre'].mean():.4f}  (makale: 0.688)")
print(f"    Recall    ort        : {anom_only['rec'].mean():.4f}  (makale: 0.690)")
print(f"    FPR       ort        : {anom_only['fpr'].mean():.4f}  (makale: 0.215)")

# Kategori
print(f"\n{'='*100}")
print(f"  KATEGORİ BAZLI (NAB F1 ile)")
print(f"{'='*100}")
print(f"{'Kategori':<18} {'n':>3} {'Strict':>8} {'NAB':>8} {'MAE':>7}  {'Mak.F1':>7}")
print("-"*70)
for cat_name in CAT_MAP.values():
    sub=rdf[rdf["category"]==cat_name]
    if len(sub)==0: continue
    pref=PAPER_CAT.get(cat_name,{})
    mf1=f"{pref['f1']:.3f}" if pref.get("f1") else "  -  "
    print(f"  {cat_name:<16} {len(sub):>3} {sub['f1'].mean():>8.4f} "
          f"{sub['nab_f1'].mean():>8.4f} {sub['mae'].mean():>7.4f}  {mf1:>7}")

rdf.to_csv(os.path.join(NAB_ROOT,"lstm_results_v12.csv"),index=False)


# ══════════════════════════════════════════════════════════════════
# GRAFİK 1: Train/Val Loss
# ══════════════════════════════════════════════════════════════════
anom=[r for r in results if r["test_anom"]>0]
if len(anom) > 0:
    cols=3; rows=(len(anom)+2)//3
    fig1,axes=plt.subplots(rows,cols,figsize=(cols*5.5,rows*3.5))
    if rows*cols == 1: axes = [axes]
    else: axes = axes.flatten()
    for idx,res in enumerate(anom):
        ax=axes[idx]
        vs=pd.Series(res["_val_h"]).rolling(3,min_periods=1).mean().values
        ep=range(1,len(res["_tr_h"])+1)
        ax.plot(ep,res["_tr_h"],color="#2196F3",lw=2,label="Train")
        ax.plot(ep,vs,"#FF9800",lw=2,label="Val (smooth)")
        ax.axvline(res["best_ep"],color="green",ls="--",lw=1.5,label=f"Best ep{res['best_ep']}")
        stl=f"STL✓p{res['period']}" if res["stl"] else "STL✗"
        fname=res["key"].split("/")[-1].replace(".csv","")[:24]
        ax.set_title(f"{fname}  [{stl}]\n"
                     f"F1={res['f1']:.3f}  Pre={res['pre']:.3f}  Rec={res['rec']:.3f}  σ_e={res['sig_e']:.2f}",
                     fontsize=7.5,fontweight="bold")
        ax.set_xlabel("Epoch",fontsize=7); ax.set_ylabel("MSE Loss",fontsize=7)
        ax.tick_params(labelsize=7); ax.legend(fontsize=6); ax.grid(True,alpha=0.3)
        ax.text(0.02,0.98,res["category"],transform=ax.transAxes,
                fontsize=6,va="top",fontweight="bold",
                bbox=dict(facecolor="white",alpha=0.7,edgecolor="#757575",boxstyle="round,pad=0.2"))
    for idx in range(len(anom),len(axes)): axes[idx].set_visible(False)
    fig1.suptitle(f"v12 — Algorithm 1 Birebir (τ={TAU} sabit, stats TÜM seriden)\n"
                  "Karami et al. (2025) | 2×LSTM(64) | STL ACF≥0.3",
                  fontsize=12,fontweight="bold")
    fig1.tight_layout()
    fig1.savefig(os.path.join(NAB_ROOT,"viz1_loss_v12.png"),dpi=130,bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════
# GRAFİK 2: Diagnostic
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax = axes[0]
diag_order = ["no_anom_test_clean", "no_anom_test_FP", "missed_all",
              "all_fp", "partial_miss", "low_f1", "mid_f1", "high_f1"]
diag_colors = {
    "no_anom_test_clean" : "#4CAF50", "no_anom_test_FP"    : "#FF9800",
    "missed_all"         : "#F44336", "all_fp"             : "#E91E63",
    "partial_miss"       : "#FFC107", "low_f1"             : "#03A9F4",
    "mid_f1"             : "#2196F3", "high_f1"            : "#0D47A1",
}
counts = [diag_counts.get(d, 0) for d in diag_order]
colors = [diag_colors[d] for d in diag_order]
bars = ax.barh(diag_order, counts, color=colors, alpha=0.85, edgecolor="white")
for bar, c in zip(bars, counts):
    if c > 0:
        ax.text(c + 0.3, bar.get_y() + bar.get_height()/2,
                f"{c} ({100*c/total:.0f}%)", va="center", fontsize=9, fontweight="bold")
ax.set_xlabel("Dosya Sayısı", fontsize=11)
ax.set_title(f"v12 Teşhis Dağılımı (Algorithm 1 birebir)\n"
             f"Toplam {total} dosya, τ={TAU} sabit, stats={{e_t}}",
             fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x"); ax.invert_yaxis()

ax = axes[1]
cat_list = list(CAT_MAP.values())
cats_short = [c.replace("Art. ","Art.") for c in cat_list]
strict_f1 = [rdf[rdf["category"]==c]["f1"].mean() if len(rdf[rdf["category"]==c])>0 else 0 for c in cat_list]
nab_f1_cat = [rdf[rdf["category"]==c]["nab_f1"].mean() if len(rdf[rdf["category"]==c])>0 else 0 for c in cat_list]
mak_f1 = [PAPER_CAT.get(c,{}).get("f1") or 0 for c in cat_list]

x = np.arange(len(cat_list)); w = 0.27
ax.bar(x - w, strict_f1, w, color="#E57373", alpha=0.85, label="Strict F1 (bizim)")
ax.bar(x, nab_f1_cat, w, color="#42A5F5", alpha=0.85, label="NAB F1 (bizim)")
ax.bar(x + w, mak_f1, w, color="#B0BEC5", alpha=0.6, hatch="///",
       edgecolor="black", lw=1, label="Makale F1")
ax.set_xticks(x); ax.set_xticklabels(cats_short, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Ortalama F1", fontsize=11); ax.set_ylim(0, 1.0)
ax.axhline(PAPER_OVERALL["f1"], color="red", ls="--", lw=1.5, alpha=0.6, label=f"Mak.ort: {PAPER_OVERALL['f1']}")
ax.set_title("Strict F1 vs NAB F1 vs Makale\n(v12: stats={e_t})",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="upper left"); ax.grid(True, alpha=0.3, axis="y")
for xi, (sf, nf) in enumerate(zip(strict_f1, nab_f1_cat)):
    ax.text(xi-w, sf+0.01, f"{sf:.2f}", ha="center", fontsize=7)
    ax.text(xi, nf+0.01, f"{nf:.2f}", ha="center", fontsize=7, fontweight="bold")

plt.suptitle(f"v12 — Makale Algorithm 1 Birebir (μ_e, σ_e ← ComputeStats({{e_t}}))",
             fontsize=13, fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(NAB_ROOT,"viz_diagnostic_v12.png"), dpi=150, bbox_inches="tight")
plt.show()
print(f"\nv12 grafikler kaydedildi.")
