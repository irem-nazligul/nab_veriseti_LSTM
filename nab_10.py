"""
NAB — LSTM Anomali Tespiti (Makale BİREBİR UYUMLU — v10)
Karami et al. (2025) arXiv:2510.11141

v9 → v10 tek değişiklik:
  - τ = 3.0 SABİT (Eq.14, k=3 sigma kuralı — makaleye birebir)
  - Val-opt mekanizması KAPALI
  - Tüm 58 dosya için tek threshold

Makale Algorithm 1 satır 23: a_t = 𝟙(z_t > τ), τ = 3
Makale Eq.14: |r_j - μ_r| / σ_r > k, k = 3

Önceki tüm düzeltmeler korunuyor:
  ✓ STL ACF≥0.3 koşullu (Algorithm 1 satır 6)
  ✓ Train residual istatistikleri (Algorithm 1 satır 20)
  ✓ Z-score normalize (Eq.2)
  ✓ 2×LSTM(64), w=50, dropout=0.2 (Section III-C3)
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
# PARAMETRELER (Makale Section III birebir)
# ══════════════════════════════════════════════════════════════════
NAB_ROOT     = r"d:\NAB"
WINDOWS_PATH = os.path.join(NAB_ROOT, "labels", "combined_windows.json")

TRAIN_R, VAL_R = 0.70, 0.15      # Eq.1
SEQ_LEN   = 50                   # w=50 (Algorithm 1)
EPOCHS    = 30                   # Section IV-A
HIDDEN    = 64                   # Section III-C3
LAYERS    = 2                    # Section III-C3
DROPOUT   = 0.2                  # Section III-C3
LR        = 1e-3                 # Section III-C3
BATCH     = 32                   # Section III-C3
GRAD_CLIP = 1.0                  # Section III-C3
PATIENCE  = 5                    # Section III-C3
ACF_THR   = 0.3                  # Section III-B

TAU       = 3.0                  # Eq.14 — SABİT, makale birebir

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
print(f"Model  : {LAYERS}×LSTM({HIDDEN}), w={SEQ_LEN}, lr={LR}, patience={PATIENCE}")
print(f"STL    : ACF≥{ACF_THR} ise aktif (tüm seri)")
print(f"τ      : {TAU} SABİT (Makale Eq.14 birebir)")
print(f"Stats  : sadece train residual'ından (Algorithm 1 satır 20)\n")


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


def compute_f1(y_true, y_pred):
    y_true = np.asarray(y_true, np.int8)
    y_pred = np.asarray(y_pred, np.int8)
    tp = int(((y_pred==1)&(y_true==1)).sum())
    fp = int(((y_pred==1)&(y_true==0)).sum())
    fn = int(((y_pred==0)&(y_true==1)).sum())
    tn = int(((y_pred==0)&(y_true==0)).sum())
    pre = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1  = 2*pre*rec/(pre+rec) if (pre+rec)>0 else 0.0
    fpr = fp/(fp+tn) if (fp+tn)>0 else 0.0
    return dict(f1=round(f1,4), pre=round(pre,4), rec=round(rec,4),
                fpr=round(fpr,4), tp=tp, fp=fp, fn=fn, tn=tn)


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

    n=len(df); i1=int(n*TRAIN_R); i2=int(n*(TRAIN_R+VAL_R))
    raw_tr=df["value"].values[:i1].astype(np.float32)
    raw_va=df["value"].values[i1:i2].astype(np.float32)
    raw_te=df["value"].values[i2:].astype(np.float32)
    y_labels  =df["label"].values[i2:].astype(np.int8)

    dt_min = df["timestamp"].diff().median().total_seconds()/60
    period  = get_period(raw_tr, dt_min)

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

    err_tr=np.abs(y_tr-infer(model,X_tr))
    y_pred_out = infer(model,X_te)
    err_te=np.abs(y_te-y_pred_out)

    # Algorithm 1 satır 20: stats SADECE train'den
    mu=err_tr.mean(); sig=err_tr.std()+1e-10
    z_te=(err_te-mu)/sig

    # Algorithm 1 satır 23: τ=3 SABİT (Eq.14)
    preds=(np.abs(z_te)>TAU).astype(np.int8)

    mae =float(np.mean(np.abs(y_te-y_pred_out)))
    rmse=float(np.sqrt(np.mean((y_te-y_pred_out)**2)))
    pcc =float(np.corrcoef(y_te,y_pred_out)[0,1]) if y_te.std()>1e-8 and y_pred_out.std()>1e-8 else 0.0
    det =compute_f1(y_labels,preds)

    cat_folder=rel_key.split("/")[0]
    category=CAT_MAP.get(cat_folder,"Unknown")

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return dict(
        key=rel_key, category=category,
        n=n, is_anomaly=bool(y_labels.sum()>0), n_anomaly=int(y_labels.sum()),
        stl=stl_ok, period=period,
        best_ep=best_ep, mae=round(mae,4), rmse=round(rmse,4), pcc=round(pcc,4),
        f1=det["f1"], pre=det["pre"], rec=det["rec"], fpr=det["fpr"],
        tp=det["tp"], fp=det["fp"], fn=det["fn"], tn=det["tn"],
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

print(f"Toplam CSV: {len(csv_files)}"); print("="*82)
results=[]
for path,key in csv_files:
    cat=CAT_MAP.get(key.split("/")[0],"?")
    print(f"  [{cat:<16}] {key.split('/')[-1][:38]:<38}",end=" ",flush=True)
    try:
        res=run_pipeline(path,key,windows)
        if res is None: print("ATLANDI"); continue
        results.append(res)
        stl=f"STL✓p{res['period']}" if res["stl"] else "STL✗"
        print(f"F1={res['f1']:.3f}  MAE={res['mae']:.3f}  Pre={res['pre']:.3f}  Rec={res['rec']:.3f}  {stl}")
    except Exception as e: print(f"HATA: {e}")


# ══════════════════════════════════════════════════════════════════
# ÖZEL DOSYALAR
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*82}")
print(f"  SEÇİLEN ÖZEL DOSYALAR İÇİN LSTM PERFORMANS METRİKLERİ")
print(f"{'='*82}")
print(f"{'Dosya Adı':<42} {'F1':>6} {'MAE':>6} {'RMSE':>6} {'R2':>7}")
print("-"*82)
target_keywords = ["art_flatline", "traveltime_387", "nyc_taxi", "machine_temperature_system_failure"]
for res in results:
    lower_key = res["key"].lower()
    if any(kw in lower_key for kw in target_keywords):
        y_true_series = res["_y_te"]; y_pred_series = res["_y_pred"]
        rmse_val = float(np.sqrt(np.mean((y_true_series - y_pred_series)**2)))
        ss_res = np.sum((y_true_series - y_pred_series) ** 2)
        ss_tot = np.sum((y_true_series - np.mean(y_true_series)) ** 2)
        r2_val = float(1 - (ss_res / ss_tot)) if ss_tot > 1e-8 else 0.0
        short_name = res["key"].split("/")[-1]
        print(f"  {short_name:<40} {res['f1']:>6.3f} {res['mae']:>6.3f} {rmse_val:>6.3f} {r2_val:>7.3f}")
print(f"{'='*82}\n")


# ══════════════════════════════════════════════════════════════════
# GENEL ÖZET
# ══════════════════════════════════════════════════════════════════
rdf   =pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")} for r in results])
rdf_cl=rdf[rdf["mae"]<5.0]

print(f"{'='*82}")
print(f"Toplam:{len(rdf)}  STL:{int(rdf['stl'].sum())}  "
      f"Anom:{int(rdf['is_anomaly'].sum())}  Normal:{int((~rdf['is_anomaly']).sum())}")
print(f"τ      : {TAU} (tüm dosyalar için sabit — Makale Eq.14)")

# Detection genel ortalamaları (Table II ile karşılaştırma)
rdf_anom = rdf[rdf["is_anomaly"]]
print(f"\n── Detection (Makale Table II) ────────────────────────────")
print(f"  Precision : {rdf_anom['pre'].mean():.4f}  (makale: 0.688)")
print(f"  Recall    : {rdf_anom['rec'].mean():.4f}  (makale: 0.690)")
print(f"  F1        : {rdf_anom['f1'].mean():.4f}  (makale: 0.688)")
print(f"  FPR       : {rdf_anom['fpr'].mean():.4f}  (makale: 0.215)")

print(f"\n── Genel Forecasting (Makale Table I) ─────────────────────")
for k,ref in [("mae",PAPER_OVERALL["mae"]),("rmse",PAPER_OVERALL["rmse"]),("pcc",PAPER_OVERALL["pcc"])]:
    print(f"  {k.upper():<5} tüm:{rdf[k].mean():.4f}  temiz:{rdf_cl[k].mean():.4f}  mak:{ref}")

print(f"\n── Genel F1 (58 dosya) ────────────────────────────────────")
print(f"  Ortalama: {rdf['f1'].mean():.4f}  (makale: {PAPER_OVERALL['f1']})")

print(f"\n── Kategori Bazlı (Makale Table III) ─────────────────────")
print(f"{'Kategori':<18} {'n':>3} {'F1':>7} {'MAE':>7} {'PCC':>7}  {'Mak.F1':>7}")
print("-"*68)
for cat_name in CAT_MAP.values():
    sub=rdf[rdf["category"]==cat_name]
    if len(sub)==0: continue
    pref=PAPER_CAT.get(cat_name,{})
    mf1=f"{pref['f1']:.3f}" if pref.get("f1") else "  -  "
    print(f"  {cat_name:<16} {len(sub):>3} {sub['f1'].mean():>7.4f} "
          f"{sub['mae'].mean():>7.4f} {sub['pcc'].mean():>7.4f}  {mf1:>7}")

rdf.to_csv(os.path.join(NAB_ROOT,"lstm_results_v10.csv"),index=False)


# ══════════════════════════════════════════════════════════════════
# GRAFİK 1: Train/Val Loss
# ══════════════════════════════════════════════════════════════════
anom=[r for r in results if r["is_anomaly"]]
cols=3; rows=(len(anom)+2)//3
fig1,axes=plt.subplots(rows,cols,figsize=(cols*5.5,rows*3.5))
axes=axes.flatten()
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
                 f"F1={res['f1']:.3f}  MAE={res['mae']:.3f}  PCC={res['pcc']:.3f}  τ={TAU}",
                 fontsize=7.5,fontweight="bold")
    ax.set_xlabel("Epoch",fontsize=7); ax.set_ylabel("MSE Loss",fontsize=7)
    ax.tick_params(labelsize=7); ax.legend(fontsize=6); ax.grid(True,alpha=0.3)
    ax.text(0.02,0.98,res["category"],transform=ax.transAxes,
            fontsize=6,va="top",fontweight="bold",
            bbox=dict(facecolor="white",alpha=0.7,edgecolor="#757575",boxstyle="round,pad=0.2"))
for idx in range(len(anom),len(axes)): axes[idx].set_visible(False)
fig1.suptitle(f"Train vs Validation Loss — v10 (Makale Birebir, τ={TAU} Sabit)\n"
              "Karami et al. (2025) | 2×LSTM(64) | STL ACF≥0.3 | Train-residual stats",
              fontsize=12,fontweight="bold")
fig1.tight_layout()
fig1.savefig(os.path.join(NAB_ROOT,"viz1_loss_v10.png"),dpi=130,bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════
# GRAFİK 2: Kategori Analizi
# ══════════════════════════════════════════════════════════════════
fig2=plt.figure(figsize=(18,12))
gs=gridspec.GridSpec(2,3,figure=fig2,hspace=0.45,wspace=0.35)

cat_list=list(CAT_MAP.values())
cat_data={cn:rdf[rdf["category"]==cn] for cn in cat_list}

ax1=fig2.add_subplot(gs[0,0])
cats_short=[c.replace("Art. ","Art.") for c in cat_list]
our_f1=[cat_data[c]["f1"].mean() if len(cat_data[c])>0 else 0 for c in cat_list]
mak_f1=[PAPER_CAT.get(c,{}).get("f1") for c in cat_list]
x=np.arange(len(cat_list)); w=0.35
bars1=ax1.bar(x-w/2,our_f1,w,color="#2196F3",alpha=0.85,label="Bizim")
ax1.bar(x+w/2,[v if v else 0 for v in mak_f1],w,
        color="#B0BEC5",alpha=0.5,edgecolor="black",lw=1.5,hatch="///",label="Makale")
ax1.set_xticks(x); ax1.set_xticklabels(cats_short,rotation=35,ha="right",fontsize=8)
ax1.set_ylabel("Ortalama F1"); ax1.set_ylim(0,1.0)
ax1.set_title("Kategori Bazlı F1\n(Mavi=Bizim, Gri Çizgili=Makale)",fontweight="bold")
ax1.legend(fontsize=8); ax1.grid(True,alpha=0.3,axis="y")
ax1.axhline(PAPER_OVERALL["f1"],color="red",ls="--",lw=1.5,alpha=0.6)
for bar,v in zip(bars1,our_f1):
    ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,
             f"{v:.2f}",ha="center",fontsize=7,fontweight="bold")

ax2=fig2.add_subplot(gs[0,1])
our_mae=[cat_data[c]["mae"].mean() if len(cat_data[c])>0 else 0 for c in cat_list]
mak_mae=[PAPER_CAT.get(c,{}).get("mae") for c in cat_list]
ax2.bar(x-w/2,[min(v,2) for v in our_mae],w,color="#2196F3",alpha=0.85,label="Bizim")
ax2.bar(x+w/2,[v if v else 0 for v in mak_mae],w,
        color="#B0BEC5",alpha=0.5,edgecolor="black",lw=1.5,hatch="///",label="Makale")
ax2.set_xticks(x); ax2.set_xticklabels(cats_short,rotation=35,ha="right",fontsize=8)
ax2.set_ylabel("Ortalama MAE"); ax2.set_ylim(0,1.5)
ax2.set_title("Kategori Bazlı MAE",fontweight="bold")
ax2.legend(fontsize=8); ax2.grid(True,alpha=0.3,axis="y")
ax2.axhline(PAPER_OVERALL["mae"],color="blue",ls="--",lw=1.5,alpha=0.6)

ax3=fig2.add_subplot(gs[0,2])
for cn in cat_list:
    sub=cat_data[cn]
    if len(sub)==0: continue
    ax3.scatter([cn.replace("Art. ","Art.")]*len(sub),sub["f1"],
                color="#1E88E5",alpha=0.6,s=50,zorder=3)
    ax3.plot([cn.replace("Art. ","Art.")]*2,
             [sub["f1"].min(),sub["f1"].max()],
             color="#1E88E5",lw=2,alpha=0.4)
    ax3.scatter(cn.replace("Art. ","Art."),sub["f1"].mean(),
                color="#0D47A1",s=150,marker="D",zorder=5,edgecolors="black",lw=1)
ax3.axhline(PAPER_OVERALL["f1"],color="red",ls="--",lw=2,label=f"Makale: {PAPER_OVERALL['f1']}")
ax3.axhline(rdf["f1"].mean(),color="navy",ls=":",lw=2,label=f"Bizim: {rdf['f1'].mean():.3f}")
ax3.set_xticklabels(ax3.get_xticklabels(),rotation=35,ha="right",fontsize=8)
ax3.set_ylabel("F1 Skoru"); ax3.set_ylim(-0.05,1.05)
ax3.set_title("Kategori İçi F1 Dağılımı",fontweight="bold")
ax3.legend(fontsize=8); ax3.grid(True,alpha=0.3,axis="y")

ax4=fig2.add_subplot(gs[1,:2])
srt=sorted(results,key=lambda r:r["f1"])
fnames_full=[r["key"].split("/")[-1].replace(".csv","")[:30] for r in srt]
f1s=[r["f1"] for r in srt]
bars=ax4.barh(range(len(f1s)),f1s,color="#1E88E5",alpha=0.85,edgecolor="white",height=0.7)
ax4.axvline(PAPER_OVERALL["f1"],color="red",ls="--",lw=2,label=f"Makale: {PAPER_OVERALL['f1']}")
ax4.axvline(rdf["f1"].mean(),color="navy",ls=":",lw=2,label=f"Bizim ort: {rdf['f1'].mean():.3f}")
ax4.set_yticks(range(len(fnames_full))); ax4.set_yticklabels(fnames_full,fontsize=5.5)
ax4.set_xlabel("F1 Skoru",fontsize=10)
ax4.set_title("Dosya Bazlı F1 — 58 Dosya",fontweight="bold")
ax4.legend(fontsize=9, loc="lower right")
for bar,v in zip(bars,f1s):
    if v>=0.005:
        ax4.text(v+0.005,bar.get_y()+bar.get_height()/2,
                 f"{v:.2f}",va="center",fontsize=5.5)

ax5=fig2.add_subplot(gs[1,2])
ax5.axis("off")
tbl_rows=[]
for cn in cat_list:
    sub=cat_data[cn]
    if len(sub)==0: continue
    pref=PAPER_CAT.get(cn,{})
    mf1=f"{pref['f1']:.3f}" if pref.get("f1") else "  -"
    mmae=f"{pref['mae']:.3f}" if pref.get("mae") else "  -"
    tbl_rows.append([cn.replace("Art. ","Art."),str(len(sub)),
                     f"{sub['f1'].mean():.3f}",mf1,
                     f"{sub['mae'].mean():.3f}",mmae])
tbl_rows.append(["─"*12,"─"*2,"─"*5,"─"*5,"─"*5,"─"*5])
tbl_rows.append(["GENEL",str(len(rdf)),
                 f"{rdf['f1'].mean():.3f}",f"{PAPER_OVERALL['f1']}",
                 f"{rdf_cl['mae'].mean():.3f}",f"{PAPER_OVERALL['mae']}"])
col_labels=["Kategori","n","F1","Mak.F1","MAE","Mak.M"]
tbl=ax5.table(cellText=tbl_rows,colLabels=col_labels,loc="center",cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1,1.6)
for j in range(len(col_labels)):
    tbl[(0,j)].set_facecolor("#1565C0")
    tbl[(0,j)].set_text_props(color="white",fontweight="bold")
for i in range(1, len(cat_list)+1):
    tbl[(i,0)].set_facecolor("#F5F5F5")
for j in range(len(col_labels)):
    tbl[(len(cat_list)+2,j)].set_facecolor("#E3F2FD")
    tbl[(len(cat_list)+2,j)].set_text_props(fontweight="bold")
ax5.set_title("Kategori Özet Tablosu",fontweight="bold",fontsize=10)

fig2.suptitle(f"LSTM — NAB v10 (Makale Birebir, τ={TAU} Sabit Eq.14)\n"
              "Karami et al. (2025) | Train-residual stats | 7 Kategori",
              fontsize=13,fontweight="bold")
fig2.savefig(os.path.join(NAB_ROOT,"viz2_category_v10.png"),dpi=150,bbox_inches="tight")
plt.show()
print(f"\nTüm grafikler {NAB_ROOT} klasörüne kaydedildi.")
