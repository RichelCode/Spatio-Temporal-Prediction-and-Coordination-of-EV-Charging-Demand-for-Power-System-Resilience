#!/usr/bin/env python3
# =============================================================================
# demo_export/export_bundle.py
#
# Standalone exporter for a small, self-contained demo bundle.
#
# Run this ON PAPERSPACE (where the trained weights and the dataset live). It
# produces:
#     demo_export/bundle.npz      - all numeric arrays (a few MB)
#     demo_export/metadata.json   - human-readable description + metrics
#
# For ~6 representative stations (station 308511 is always included) it saves:
#   * ground-truth flow (with timestamps)                    [vehicles/hour]
#   * GraphWaveNet-GRU-LSTM predictions at 12/24/48/72h       [vehicles/hour]
#   * Random Forest predictions at 12/24/48/72h               [vehicles/hour]
#   * MAE and RMSE per model per horizon (per station)        [vehicles/hour]
#   * EV load profile P_EV(t)=P_max*q(t)/max(q), actual+fcst  [kW]
#   * 72h forecasts under 0/10/20/30% sensor outage, both     [vehicles/hour]
#
# This script only READS your trained model + dataset. It never touches your
# training / model code, and writes only inside demo_export/.
#
# It is a distilled, standalone copy of the logic in
# `Paper_Robustness_Peaks_FINAL.ipynb` (model classes, checkpoint loader,
# windowing, RF baseline, and the sensor-outage evaluator).
# =============================================================================

import os
import re
import sys
import json
import math
from pathlib import Path

import numpy as np

# -----------------------------------------------------------------------------
# 1) CONSTANTS  ---  EDIT THESE TWO PATHS FOR YOUR PAPERSPACE MACHINE
# -----------------------------------------------------------------------------

# Path to the trained GraphWaveNet-GRU-LSTM checkpoint.
# Accepts EITHER the `best.pt` file directly, OR the run directory that
# contains `best.pt` (e.g. artifacts/runs/<timestamp>_..._GraphWaveNet_GRU_LSTM).
CHECKPOINT_PATH = "artifacts/runs/CHANGE_ME_GraphWaveNet_GRU_LSTM/best.pt"

# Path to the strict graph dataset artifact produced by your data pipeline.
DATA_PATH = "artifacts/pems_graph_dataset_strict.npz"

# ---- Output ----
OUT_DIR = Path(__file__).resolve().parent          # demo_export/
BUNDLE_NPZ = OUT_DIR / "bundle.npz"
METADATA_JSON = OUT_DIR / "metadata.json"

# ---- Station selection ----
REQUIRED_STATION = "308511"     # always included (paper showcase station)
N_STATIONS = 6                  # total stations in the bundle
# Optional hard override: put exact station IDs here (strings) to skip the
# automatic volume-based selection, e.g. ["308511", "311903", ...].
STATIONS_OVERRIDE = None

# ---- Evaluation settings (match the paper) ----
HORIZONS = [12, 24, 48, 72]     # reported forecast horizons (hours)
OUTAGE_RATES = [0.0, 0.10, 0.20, 0.30]  # 0.0 == clean
OUTAGE_SEED = 0                 # fixed seed so the outage pattern is reproducible

# ---- EV mapping ----
P_MAX_KW = 200.0                # P_max in the P_EV(t) mapping [kW]
EV_HORIZON = 24                 # horizon (h) used for the forecast-driven EV curve

# ---- Random Forest baseline (identical to the notebooks) ----
RF_PARAMS = dict(
    n_estimators=50,
    max_depth=20,
    min_samples_leaf=5,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
)

BATCH_SIZE = 8


# -----------------------------------------------------------------------------
# 2) FRIENDLY DEPENDENCY / PATH CHECKS
# -----------------------------------------------------------------------------
def _die(msg: str):
    print("\n[export_bundle] ERROR: " + msg + "\n", file=sys.stderr)
    sys.exit(1)


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception as e:  # pragma: no cover
    _die("PyTorch is required but could not be imported (%r).\n"
         "Activate the environment that has torch installed (the same one you "
         "trained in)." % e)

try:
    from sklearn.ensemble import RandomForestRegressor
except Exception as e:  # pragma: no cover
    _die("scikit-learn is required for the Random Forest baseline (%r)." % e)

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    _die("pandas is required (%r)." % e)


def resolve_checkpoint(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_dir():
        cand = p / "best.pt"
        if not cand.exists():
            _die("CHECKPOINT_PATH is a directory but has no best.pt inside it:\n"
                 "    %s\n"
                 "Point CHECKPOINT_PATH at the run folder that contains best.pt, "
                 "or at the best.pt file itself." % p)
        return cand
    if not p.exists():
        _die("CHECKPOINT_PATH does not exist:\n    %s\n"
             "Set CHECKPOINT_PATH (top of this script) to your trained "
             "GraphWaveNet-GRU-LSTM best.pt (or its run folder) on Paperspace."
             % p)
    return p


def resolve_data(path_str: str) -> Path:
    p = Path(path_str)
    if not p.exists():
        _die("DATA_PATH does not exist:\n    %s\n"
             "Set DATA_PATH (top of this script) to your strict dataset .npz "
             "(e.g. artifacts/pems_graph_dataset_strict.npz)." % p)
    return p


CKPT = resolve_checkpoint(CHECKPOINT_PATH)
DATA = resolve_data(DATA_PATH)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[export_bundle] Device      :", DEVICE)
print("[export_bundle] Checkpoint  :", CKPT)
print("[export_bundle] Dataset     :", DATA)


# -----------------------------------------------------------------------------
# 3) LOAD DATASET ARTIFACT
#    (keys as produced by the data pipeline / used in Paper_Robustness_Peaks_FINAL)
# -----------------------------------------------------------------------------
data = np.load(DATA, allow_pickle=True)
_need = ["X", "Y", "A", "stations", "timestamps",
         "flow_mean", "flow_std", "in_len", "out_len",
         "train_starts", "test_starts"]
_missing = [k for k in _need if k not in data.files]
if _missing:
    _die("Dataset %s is missing expected keys: %s\nFound keys: %s"
         % (DATA, _missing, list(data.files)))

X_raw = data["X"].astype(np.float32)          # (T, N, F)  flow[, speed]
Y_raw = data["Y"].astype(np.float32)          # (T, N)     ground-truth flow (unscaled)
A = data["A"].astype(np.float32)              # (N, N)     adjacency
stations = np.array([str(s) for s in data["stations"]])   # (N,)
timestamps = pd.to_datetime(data["timestamps"])           # (T,)
flow_mean = data["flow_mean"].astype(np.float32)          # (N,)
flow_std = data["flow_std"].astype(np.float32)            # (N,)
speed_mean = data["speed_mean"].astype(np.float32) if "speed_mean" in data.files else None
speed_std = data["speed_std"].astype(np.float32) if "speed_std" in data.files else None

IN_LEN = int(np.array(data["in_len"]).item())
OUT_LEN = int(np.array(data["out_len"]).item())
train_starts = data["train_starts"].astype(np.int64)
test_starts = data["test_starts"].astype(np.int64)

T, N, F_in = X_raw.shape
print("[export_bundle] X_raw       :", X_raw.shape, "(T,N,F)")
print("[export_bundle] IN_LEN/OUT_LEN:", IN_LEN, OUT_LEN,
      "| #test windows:", len(test_starts))

if OUT_LEN < max(HORIZONS):
    _die("OUT_LEN=%d in the dataset is smaller than max(HORIZONS)=%d."
         % (OUT_LEN, max(HORIZONS)))
if EV_HORIZON not in HORIZONS:
    _die("EV_HORIZON=%d must be one of HORIZONS=%s." % (EV_HORIZON, HORIZONS))

HSEL = len(HORIZONS)
H_OFF = np.array([h - 1 for h in HORIZONS], dtype=np.int64)   # 0-based horizon offsets


# -----------------------------------------------------------------------------
# 4) SCALED ARRAYS + TIME FEATURES  (same recipe as the notebook)
# -----------------------------------------------------------------------------
def time_encoding(dt_index: pd.DatetimeIndex) -> np.ndarray:
    hours = dt_index.hour.values
    dow = dt_index.dayofweek.values
    return np.stack([
        np.sin(2 * np.pi * hours / 24.0), np.cos(2 * np.pi * hours / 24.0),
        np.sin(2 * np.pi * dow / 7.0),    np.cos(2 * np.pi * dow / 7.0),
    ], axis=1).astype(np.float32)


TF_all = time_encoding(pd.DatetimeIndex(timestamps))         # (T, 4)

X_scaled = X_raw.copy()
X_scaled[:, :, 0] = (X_scaled[:, :, 0] - flow_mean[None, :]) / (flow_std[None, :] + 1e-6)
if F_in > 1 and speed_mean is not None and speed_std is not None:
    X_scaled[:, :, 1] = (X_scaled[:, :, 1] - speed_mean[None, :]) / (speed_std[None, :] + 1e-6)
Y_scaled = (Y_raw - flow_mean[None, :]) / (flow_std[None, :] + 1e-6)

flow_mean_t = torch.tensor(flow_mean, dtype=torch.float32, device=DEVICE).view(1, 1, -1)
flow_std_t = torch.tensor(flow_std, dtype=torch.float32, device=DEVICE).view(1, 1, -1)


class PemsWindowDatasetTF(Dataset):
    """Sliding-window dataset. x -> (F, N, IN_LEN); y -> (OUT_LEN, N); tf -> (OUT_LEN, 4)."""
    def __init__(self, X_scaled, Y_scaled, TF_all, starts, in_len, out_len):
        self.X_scaled, self.Y_scaled, self.TF_all = X_scaled, Y_scaled, TF_all
        self.starts = starts.astype(np.int64)
        self.in_len, self.out_len = int(in_len), int(out_len)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        t = int(self.starts[idx])
        x = self.X_scaled[t:t + self.in_len].copy().astype(np.float32)
        y = self.Y_scaled[t + self.in_len:t + self.in_len + self.out_len].copy().astype(np.float32)
        tf = self.TF_all[t + self.in_len:t + self.in_len + self.out_len].copy().astype(np.float32)
        x = np.transpose(x, (2, 1, 0))       # (F, N, IN_LEN)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(tf)


test_ds = PemsWindowDatasetTF(X_scaled, Y_scaled, TF_all, test_starts, IN_LEN, OUT_LEN)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0, pin_memory=(DEVICE == "cuda"))


# -----------------------------------------------------------------------------
# 5) GRAPH SUPPORTS (adjacency -> row-normalised sparse forward/backward)
# -----------------------------------------------------------------------------
def row_normalize(A_dense, eps=1e-6):
    d = A_dense.sum(axis=1, keepdims=True)
    return A_dense / (d + eps)


def dense_to_sparse(A_dense, device):
    idx = np.nonzero(A_dense)
    return torch.sparse_coo_tensor(
        torch.tensor(np.vstack(idx), dtype=torch.long, device=device),
        torch.tensor(A_dense[idx].astype(np.float32), dtype=torch.float32, device=device),
        size=A_dense.shape, device=device,
    ).coalesce()


A_hat = A + np.eye(A.shape[0], dtype=np.float32)
supports = [dense_to_sparse(row_normalize(A_hat), DEVICE),
            dense_to_sparse(row_normalize(A_hat.T), DEVICE)]


# -----------------------------------------------------------------------------
# 6) MODEL DEFINITION  (verbatim from Paper_Robustness_Peaks_FINAL, cell 11)
# -----------------------------------------------------------------------------
class NConv(nn.Module):
    def forward(self, x, A_sp):
        B, C, Nn, Tn = x.shape
        x_r = x.permute(2, 0, 1, 3).reshape(Nn, -1)
        out = torch.sparse.mm(A_sp, x_r.float())
        out = out.reshape(Nn, B, C, Tn).permute(1, 2, 0, 3)
        return out.to(dtype=x.dtype)


class DiffusionGraphConv(nn.Module):
    def __init__(self, c_in, c_out, supports, order=1, dropout=0.0):
        super().__init__()
        self.nconv = NConv()
        self.supports = supports
        self.order = order
        self.dropout = dropout
        c_total = c_in * (1 + len(supports) * order)
        self.mlp = nn.Conv2d(c_total, c_out, kernel_size=(1, 1))

    def forward(self, x):
        out = [x]
        for A_sp in self.supports:
            x1 = self.nconv(x, A_sp)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = self.nconv(x1, A_sp)
                out.append(x1)
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return F.dropout(h, p=self.dropout, training=self.training)


class CausalConv2d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=2, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=(1, kernel_size), dilation=(1, dilation))

    def forward(self, x):
        x = F.pad(x, (self.pad, 0, 0, 0))
        return self.conv(x)


class GraphWaveNetEncoder(nn.Module):
    def __init__(self, num_nodes, in_dim, supports, residual_channels=32,
                 dilation_channels=32, skip_channels=64, end_channels=128,
                 kernel_size=2, blocks=2, layers_per_block=4, gcn_order=1, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        receptive_field = 1
        for _ in range(blocks):
            for i in range(layers_per_block):
                receptive_field += (kernel_size - 1) * (2 ** i)
        self.receptive_field = receptive_field

        self.start_conv = nn.Conv2d(in_dim, residual_channels, kernel_size=(1, 1))
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconvs = nn.ModuleList()
        for _ in range(blocks):
            for i in range(layers_per_block):
                dilation = 2 ** i
                self.filter_convs.append(CausalConv2d(residual_channels, dilation_channels, kernel_size, dilation))
                self.gate_convs.append(CausalConv2d(residual_channels, dilation_channels, kernel_size, dilation))
                self.skip_convs.append(nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1)))
                self.gconvs.append(DiffusionGraphConv(dilation_channels, residual_channels, supports, order=gcn_order, dropout=dropout))
                self.bn.append(nn.BatchNorm2d(residual_channels))
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1))

    def forward(self, x):
        if x.size(-1) < self.receptive_field:
            x = F.pad(x, (self.receptive_field - x.size(-1), 0, 0, 0))
        x = self.start_conv(x)
        skip = None
        for i in range(len(self.filter_convs)):
            residual = x
            filt = torch.tanh(self.filter_convs[i](x))
            gate = torch.sigmoid(self.gate_convs[i](x))
            x = filt * gate
            x = F.dropout(x, p=self.dropout, training=self.training)
            s = self.skip_convs[i](x)
            skip = s if skip is None else (skip + s)
            x = self.gconvs[i](x)
            x = x + residual
            x = self.bn[i](x)
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        return x


class GraphWaveNetRNN(nn.Module):
    def __init__(self, num_nodes, in_dim, out_len, supports, residual_channels=32,
                 dilation_channels=32, skip_channels=64, end_channels=128, kernel_size=2,
                 blocks=2, layers_per_block=4, gcn_order=1, dropout=0.1,
                 use_gru=False, use_lstm=False, rnn_hidden=128):
        super().__init__()
        self.out_len = out_len
        self.use_gru = use_gru
        self.use_lstm = use_lstm
        self.encoder = GraphWaveNetEncoder(
            num_nodes=num_nodes, in_dim=in_dim, supports=supports,
            residual_channels=residual_channels, dilation_channels=dilation_channels,
            skip_channels=skip_channels, end_channels=end_channels, kernel_size=kernel_size,
            blocks=blocks, layers_per_block=layers_per_block, gcn_order=gcn_order, dropout=dropout)
        self.gru = nn.GRU(input_size=end_channels, hidden_size=rnn_hidden, batch_first=True) if use_gru else None
        self.lstm = nn.LSTM(input_size=(rnn_hidden if use_gru else end_channels),
                            hidden_size=rnn_hidden, batch_first=True) if use_lstm else None
        final_dim = rnn_hidden if (use_gru or use_lstm) else end_channels
        self.time_embed = nn.Linear(4, final_dim)
        self.horizon_out = nn.Linear(final_dim, 1)

    def forward(self, x, tf_future):
        h = self.encoder(x)
        B, C, Nn, Tn = h.shape
        seq = h.permute(0, 2, 3, 1).contiguous().view(B * Nn, Tn, C)
        if self.gru is not None:
            seq, _ = self.gru(seq)
        if self.lstm is not None:
            seq, _ = self.lstm(seq)
        last = seq[:, -1, :]
        z = last.view(B, Nn, -1)
        te = self.time_embed(tf_future)
        out = F.relu(z.unsqueeze(1) + te.unsqueeze(2))
        return self.horizon_out(out).squeeze(-1)     # (B, OUT_LEN, N)


# -----------------------------------------------------------------------------
# 7) CHECKPOINT LOADER  (infers architecture from the saved state_dict)
# -----------------------------------------------------------------------------
def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, nn.Module):
        return None
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model_state_dict", "model"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                sd = ckpt_obj[k]
                if all(isinstance(v, torch.Tensor) for v in sd.values()):
                    return sd
        if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    return None


def _strip_module_prefix(sd):
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def infer_gwn_params_from_state_dict(sd, cfg, supports_len=2):
    residual_channels = int(sd["encoder.start_conv.weight"].shape[0])
    dilation_channels = int(sd["encoder.filter_convs.0.conv.weight"].shape[0])
    kernel_size = int(sd["encoder.filter_convs.0.conv.weight"].shape[3])
    skip_channels = int(sd["encoder.skip_convs.0.weight"].shape[0])
    end_channels = int(sd["encoder.end_conv_1.weight"].shape[0])

    layer_ids = sorted({int(m.group(1)) for k in sd.keys()
                        for m in [re.match(r"encoder\.filter_convs\.(\d+)\.conv\.weight", k)] if m})
    total_layers = len(layer_ids)

    blocks = cfg.get("blocks")
    layers_per_block = cfg.get("layers_per_block")
    if blocks is None or layers_per_block is None:
        if total_layers % 4 == 0:
            layers_per_block, blocks = 4, total_layers // 4
        elif total_layers % 3 == 0:
            layers_per_block, blocks = 3, total_layers // 3
        else:
            blocks, layers_per_block = 1, total_layers

    gcn_order = cfg.get("gcn_order")
    if gcn_order is None:
        c_total = int(sd["encoder.gconvs.0.mlp.weight"].shape[1])
        gcn_order = max(1, int(round((c_total / dilation_channels - 1.0) / supports_len)))

    use_gru = any(k.startswith("gru.") for k in sd.keys())
    use_lstm = any(k.startswith("lstm.") for k in sd.keys())
    if use_lstm:
        rnn_hidden = int(sd["lstm.weight_ih_l0"].shape[0] // 4)
    elif use_gru:
        rnn_hidden = int(sd["gru.weight_ih_l0"].shape[0] // 3)
    else:
        rnn_hidden = end_channels

    return dict(residual_channels=residual_channels, dilation_channels=dilation_channels,
                skip_channels=skip_channels, end_channels=end_channels, kernel_size=kernel_size,
                blocks=int(blocks), layers_per_block=int(layers_per_block), gcn_order=int(gcn_order),
                dropout=float(cfg.get("dropout", 0.0)),
                use_gru=bool(use_gru), use_lstm=bool(use_lstm), rnn_hidden=int(rnn_hidden))


def load_proposed_model(ckpt_path: Path):
    cfg = {}
    cfg_file = ckpt_path.parent / "config.json"
    if cfg_file.exists():
        try:
            cfg = json.load(open(cfg_file))
        except Exception:
            cfg = {}
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, nn.Module):
        ckpt.eval()
        return ckpt.to(DEVICE), {"_note": "full nn.Module checkpoint"}
    sd = _extract_state_dict(ckpt)
    if sd is None:
        _die("Unrecognised checkpoint format in %s (expected a state_dict or nn.Module)." % ckpt_path)
    sd = _strip_module_prefix(sd)
    params = infer_gwn_params_from_state_dict(sd, cfg, supports_len=len(supports))
    model = GraphWaveNetRNN(num_nodes=N, in_dim=F_in, out_len=OUT_LEN, supports=supports, **params)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print("[export_bundle] WARNING missing keys (first 10):", list(missing)[:10])
    if unexpected:
        print("[export_bundle] WARNING unexpected keys (first 10):", list(unexpected)[:10])
    if not params.get("use_gru") or not params.get("use_lstm"):
        print("[export_bundle] NOTE: checkpoint has use_gru=%s use_lstm=%s "
              "(expected both True for GraphWaveNet-GRU-LSTM)."
              % (params.get("use_gru"), params.get("use_lstm")))
    model.eval()
    return model.to(DEVICE), params


proposed_model, proposed_params = load_proposed_model(CKPT)
print("[export_bundle] Inferred model params:", proposed_params)

# Forward sanity check
with torch.no_grad():
    xb, yb, tfb = next(iter(test_loader))
    yhat = proposed_model(xb.to(DEVICE), tfb.to(DEVICE))
if tuple(yhat.shape) != tuple(yb.shape):
    _die("Forward-shape check failed: got %s, expected %s. The checkpoint likely "
         "does not match this dataset (N=%d, OUT_LEN=%d)."
         % (tuple(yhat.shape), tuple(yb.shape), N, OUT_LEN))
print("[export_bundle] Forward check passed:", tuple(yhat.shape))


# -----------------------------------------------------------------------------
# 8) STATION SELECTION  (auto by traffic-volume spread; 308511 always included)
# -----------------------------------------------------------------------------
# Region of the timeline actually covered by test-window targets, used both for
# ranking stations and (later) normalising the EV curve.
test_target_idx = np.unique(
    (test_starts[:, None] + IN_LEN + H_OFF[None, :]).reshape(-1)
)
test_target_idx = test_target_idx[test_target_idx < T]
mean_flow = Y_raw[test_target_idx].mean(axis=0)     # (N,) mean test-period flow per station


def resolve_stations():
    if STATIONS_OVERRIDE:
        idxs = []
        for sid in STATIONS_OVERRIDE:
            hit = np.where(stations == str(sid))[0]
            if len(hit) == 0:
                _die("STATIONS_OVERRIDE station %r not found in dataset. "
                     "Available example IDs: %s ..." % (sid, list(stations[:8])))
            idxs.append(int(hit[0]))
        return idxs

    order = np.argsort(mean_flow)                   # ascending by volume
    qs = np.linspace(0.90, 0.10, N_STATIONS - 1)    # high -> low spread
    picks = [int(order[int(round(q * (len(order) - 1)))]) for q in qs]

    req = np.where(stations == REQUIRED_STATION)[0]
    if len(req) == 0:
        print("[export_bundle] WARNING: required station %s not in dataset; "
              "using volume-spread picks only." % REQUIRED_STATION)
    else:
        req_idx = int(req[0])
        if req_idx not in picks:
            picks = [req_idx] + picks

    # de-duplicate, preserve order, trim to N_STATIONS
    seen, uniq = set(), []
    for i in picks:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    # top up from the volume order if dedup left us short (guarantees N_STATIONS
    # whenever the dataset has that many nodes)
    for cand in order:
        if len(uniq) >= N_STATIONS:
            break
        cand = int(cand)
        if cand not in seen:
            seen.add(cand)
            uniq.append(cand)
    return uniq[:N_STATIONS]


sel_nodes = resolve_stations()
sel_ids = [stations[i] for i in sel_nodes]
print("[export_bundle] Selected stations (node idx -> id, mean flow):")
for i in sel_nodes:
    print("    %5d -> %-8s  mean_flow=%8.1f veh/h" % (i, stations[i], mean_flow[i]))
sel_nodes_arr = np.array(sel_nodes, dtype=np.int64)
n_sel = len(sel_nodes)


# -----------------------------------------------------------------------------
# 9) OUTAGE MASK HELPER  (verbatim behaviour from the notebook)
# -----------------------------------------------------------------------------
def make_fixedcount_node_masks(num_samples, num_nodes, p, seed):
    rng = np.random.default_rng(seed)
    m = int(round(p * num_nodes))
    m = max(m, 1) if p > 0 else 0
    out = np.zeros((num_samples, num_nodes), dtype=bool)
    if m == 0:
        return out
    for i in range(num_samples):
        out[i, rng.choice(num_nodes, size=m, replace=False)] = True
    return out


# -----------------------------------------------------------------------------
# 10) GRAPHWAVENET-GRU-LSTM FORWARD PASS OVER TEST SET
#     Returns de-scaled predictions for the selected nodes only.
#     node_mask: optional (S_test, N) bool -> zeroes those input nodes.
# -----------------------------------------------------------------------------
@torch.inference_mode()
def gwn_predict(node_mask=None, keep_full=True):
    proposed_model.eval()
    keep_dim = OUT_LEN if keep_full else 1
    S = len(test_ds)
    out = np.zeros((S, keep_dim, n_sel), dtype=np.float32)
    sel_t = torch.tensor(sel_nodes_arr, device=DEVICE)
    offset = 0
    for xb, yb, tfb in test_loader:
        B = xb.shape[0]
        xb = xb.clone().to(DEVICE, non_blocking=True)
        tfb = tfb.to(DEVICE, non_blocking=True)
        if node_mask is not None:
            bm = torch.from_numpy(node_mask[offset:offset + B]).to(DEVICE)
            xb = xb.masked_fill(bm[:, None, :, None], 0.0)
        pred_scaled = proposed_model(xb, tfb)                       # (B, OUT_LEN, N)
        pred_u = pred_scaled * flow_std_t + flow_mean_t            # de-scale
        pred_sel = pred_u.index_select(2, sel_t)                   # (B, OUT_LEN, n_sel)
        if not keep_full:
            pred_sel = pred_sel[:, 71:72, :]                       # 72h only (index 71)
        out[offset:offset + B] = pred_sel.detach().cpu().numpy()
        offset += B
    return out


print("[export_bundle] Running GraphWaveNet-GRU-LSTM clean forward pass ...")
gwn_full = gwn_predict(node_mask=None, keep_full=True)     # (S, OUT_LEN, n_sel), de-scaled


# -----------------------------------------------------------------------------
# 11) RANDOM FOREST BASELINE  (one RF per selected node; identical features)
# -----------------------------------------------------------------------------
HIST_DIM = IN_LEN * F_in


def node_features_and_targets(node, starts):
    Xn = X_scaled[:, node, :]                                  # (T, F)
    win = np.lib.stride_tricks.sliding_window_view(Xn, window_shape=IN_LEN, axis=0)
    X_hist = win[starts].reshape(len(starts), -1)             # (len, F*IN_LEN)
    idx = starts[:, None] + IN_LEN + H_OFF[None, :]           # (len, HSEL)
    X_tf = TF_all[idx].reshape(len(starts), -1)               # (len, HSEL*4)
    X_feat = np.concatenate([X_hist, X_tf], axis=1).astype(np.float32)
    y = Y_scaled[idx, node].astype(np.float32)                # (len, HSEL)
    return X_feat, y


print("[export_bundle] Fitting Random Forest baselines (per selected station) ...")
rf_models = {}
rf_full = np.zeros((len(test_starts), HSEL, n_sel), dtype=np.float32)  # de-scaled preds
for j, node in enumerate(sel_nodes):
    Xtr, ytr = node_features_and_targets(node, train_starts)
    Xte, _ = node_features_and_targets(node, test_starts)
    mdl = RandomForestRegressor(**RF_PARAMS)
    mdl.fit(Xtr, ytr)
    rf_models[node] = mdl
    pred_scaled = mdl.predict(Xte).astype(np.float32)          # (S, HSEL)
    rf_full[:, :, j] = pred_scaled * flow_std[node] + flow_mean[node]
    print("    station %-8s RF fitted." % stations[node])


# -----------------------------------------------------------------------------
# 12) GROUND TRUTH + PER-HORIZON TIMESTAMPS (from Y_raw directly)
# -----------------------------------------------------------------------------
ts_values = np.asarray(timestamps.values)         # datetime64[ns], (T,)
true_by_h, gwn_by_h, rf_by_h, ts_by_h = {}, {}, {}, {}
for hj, h in enumerate(HORIZONS):
    tgt = test_starts + IN_LEN + (h - 1)          # target index per window
    ts_by_h[h] = ts_values[tgt]                   # (S,)
    true_by_h[h] = Y_raw[tgt][:, sel_nodes_arr]   # (S, n_sel) ground-truth flow
    gwn_by_h[h] = gwn_full[:, h - 1, :]           # (S, n_sel)
    rf_by_h[h] = rf_full[:, hj, :]                # (S, n_sel)


def mae_rmse(pred, true):
    err = pred - true
    return float(np.abs(err).mean()), float(np.sqrt((err ** 2).mean()))


# per-station, per-horizon, per-model metrics
metrics = {stations[node]: {"GraphWaveNet-GRU-LSTM": {}, "RandomForest": {}}
           for node in sel_nodes}
for hj, h in enumerate(HORIZONS):
    for j, node in enumerate(sel_nodes):
        sid = stations[node]
        mae_g, rmse_g = mae_rmse(gwn_by_h[h][:, j], true_by_h[h][:, j])
        mae_r, rmse_r = mae_rmse(rf_by_h[h][:, j], true_by_h[h][:, j])
        metrics[sid]["GraphWaveNet-GRU-LSTM"][str(h)] = {"MAE": mae_g, "RMSE": rmse_g}
        metrics[sid]["RandomForest"][str(h)] = {"MAE": mae_r, "RMSE": rmse_r}


# -----------------------------------------------------------------------------
# 13) EV LOAD PROFILE  P_EV(t) = P_MAX * q(t) / max(q)   [kW]  (thesis form)
#     Evaluated on the EV_HORIZON forecast grid so actual vs forecast align.
#     Literal per-series normalisation: each curve is divided by ITS OWN maximum,
#     so both the actual and forecast EV curves peak at P_MAX_KW.
# -----------------------------------------------------------------------------
ev_ts = ts_by_h[EV_HORIZON]                                   # (S,)
ev_true_flow = true_by_h[EV_HORIZON]                          # (S, n_sel)
ev_gwn_flow = gwn_by_h[EV_HORIZON]                            # (S, n_sel)
ev_actual_ref_max = np.maximum(ev_true_flow.max(axis=0), 1e-6)    # (n_sel,) max of actual flow
ev_forecast_ref_max = np.maximum(ev_gwn_flow.max(axis=0), 1e-6)   # (n_sel,) max of forecast flow
ev_actual = (P_MAX_KW * ev_true_flow / ev_actual_ref_max[None, :]).astype(np.float32)
ev_gwn = (P_MAX_KW * ev_gwn_flow / ev_forecast_ref_max[None, :]).astype(np.float32)


# -----------------------------------------------------------------------------
# 14) SENSOR-OUTAGE 72h FORECASTS  (0/10/20/30%) for both models
# -----------------------------------------------------------------------------
H72_POS = HORIZONS.index(72)
S_test = len(test_starts)
outage_gwn_72 = np.zeros((len(OUTAGE_RATES), S_test, n_sel), dtype=np.float32)
outage_rf_72 = np.zeros((len(OUTAGE_RATES), S_test, n_sel), dtype=np.float32)
outage_true_72 = true_by_h[72]                               # (S, n_sel)
outage_ts_72 = ts_by_h[72]

for ri, p in enumerate(OUTAGE_RATES):
    print("[export_bundle] Outage forecast @72h, rate=%.0f%% ..." % (p * 100))
    mask = None if p == 0.0 else make_fixedcount_node_masks(S_test, N, p, OUTAGE_SEED)

    # --- graph model ---
    if p == 0.0:
        outage_gwn_72[ri] = gwn_full[:, 71, :]
    else:
        outage_gwn_72[ri] = gwn_predict(node_mask=mask, keep_full=False)[:, 0, :]

    # --- random forest (self-masks the node's own history where it is out) ---
    for j, node in enumerate(sel_nodes):
        Xte, _ = node_features_and_targets(node, test_starts)
        if mask is not None:
            mrows = mask[:, node]
            if np.any(mrows):
                Xte = Xte.copy()
                Xte[mrows, :HIST_DIM] = 0.0
        pred_scaled = rf_models[node].predict(Xte).astype(np.float32)   # (S, HSEL)
        outage_rf_72[ri, :, j] = pred_scaled[:, H72_POS] * flow_std[node] + flow_mean[node]

# outage MAE summary (per station, per rate, per model) for metadata
outage_summary = {stations[node]: {} for node in sel_nodes}
for ri, p in enumerate(OUTAGE_RATES):
    for j, node in enumerate(sel_nodes):
        sid = stations[node]
        mae_g, rmse_g = mae_rmse(outage_gwn_72[ri, :, j], outage_true_72[:, j])
        mae_r, rmse_r = mae_rmse(outage_rf_72[ri, :, j], outage_true_72[:, j])
        outage_summary[sid]["%d%%" % int(p * 100)] = {
            "GraphWaveNet-GRU-LSTM": {"MAE72": mae_g, "RMSE72": rmse_g},
            "RandomForest": {"MAE72": mae_r, "RMSE72": rmse_r},
        }


# -----------------------------------------------------------------------------
# 15) ASSEMBLE + WRITE BUNDLE.NPZ
# -----------------------------------------------------------------------------
npz = {
    "stations": np.array(sel_ids),
    "station_node_index": sel_nodes_arr,
    "horizons": np.array(HORIZONS, dtype=np.int64),
    "outage_rates_pct": np.array([int(p * 100) for p in OUTAGE_RATES], dtype=np.int64),
    "ev_horizon": np.int64(EV_HORIZON),
    "p_max_kw": np.float32(P_MAX_KW),
    "ev_flow_ref_max_actual": ev_actual_ref_max.astype(np.float32),      # (n_sel,)
    "ev_flow_ref_max_forecast": ev_forecast_ref_max.astype(np.float32),  # (n_sel,)
    "ev_timestamps": ev_ts,                                   # (S,) datetime64
    "ev_load_actual_kw": ev_actual,                          # (S, n_sel)
    "ev_load_forecast_kw": ev_gwn,                           # (S, n_sel)
    "outage_timestamps_72": outage_ts_72,                    # (S,) datetime64
    "outage_true_72": outage_true_72.astype(np.float32),     # (S, n_sel)
    "outage_gwn_72": outage_gwn_72,                          # (rates, S, n_sel)
    "outage_rf_72": outage_rf_72,                            # (rates, S, n_sel)
}
for h in HORIZONS:
    npz["timestamps_h%d" % h] = ts_by_h[h]                   # (S,) datetime64
    npz["true_h%d" % h] = true_by_h[h].astype(np.float32)   # (S, n_sel)
    npz["gwn_h%d" % h] = gwn_by_h[h].astype(np.float32)     # (S, n_sel)
    npz["rf_h%d" % h] = rf_by_h[h].astype(np.float32)       # (S, n_sel)
# metrics as arrays too (rows follow `stations`, cols follow `horizons`)
for model_key, tag in [("GraphWaveNet-GRU-LSTM", "gwn"), ("RandomForest", "rf")]:
    mae_arr = np.array([[metrics[sid][model_key][str(h)]["MAE"] for h in HORIZONS] for sid in sel_ids], dtype=np.float32)
    rmse_arr = np.array([[metrics[sid][model_key][str(h)]["RMSE"] for h in HORIZONS] for sid in sel_ids], dtype=np.float32)
    npz["mae_%s" % tag] = mae_arr
    npz["rmse_%s" % tag] = rmse_arr

np.savez_compressed(BUNDLE_NPZ, **npz)
size_mb = BUNDLE_NPZ.stat().st_size / 1e6
print("[export_bundle] Wrote %s (%.2f MB)" % (BUNDLE_NPZ, size_mb))


# -----------------------------------------------------------------------------
# 16) WRITE METADATA.JSON
# -----------------------------------------------------------------------------
meta = {
    "description": "Demo bundle: GraphWaveNet-GRU-LSTM vs Random Forest traffic "
                   "forecasts + EV load mapping + sensor-outage robustness, for "
                   "~6 representative PeMS District 3 stations.",
    "generated_from": {
        "checkpoint": str(CKPT),
        "dataset": str(DATA),
        "device": DEVICE,
        "model_params_inferred": {k: (int(v) if isinstance(v, (np.integer, bool, int)) else v)
                                  for k, v in proposed_params.items()},
    },
    "config": {
        "input_length_L_hours": IN_LEN,
        "forecast_horizon_H_hours": OUT_LEN,
        "reported_horizons_hours": HORIZONS,
        "outage_rates_pct": [int(p * 100) for p in OUTAGE_RATES],
        "outage_seed": OUTAGE_SEED,
        "P_max_kW": P_MAX_KW,
        "ev_horizon_hours": EV_HORIZON,
        "rf_params": RF_PARAMS,
        "num_test_windows": int(S_test),
        "num_nodes_total": int(N),
    },
    "units": {
        "flow": "vehicles/hour",
        "power": "kW",
        "timestamps": "np.datetime64[ns] (hourly)",
    },
    "stations": [
        {"station_id": stations[node], "node_index": int(node),
         "mean_test_flow_veh_h": float(mean_flow[node]),
         "ev_flow_ref_max_actual_veh_h": float(ev_actual_ref_max[j]),
         "ev_flow_ref_max_forecast_veh_h": float(ev_forecast_ref_max[j])}
        for j, node in enumerate(sel_nodes)
    ],
    "metrics_per_station_per_horizon": metrics,
    "outage_mae72_per_station_per_rate": outage_summary,
    "arrays": {
        "stations": "(6,) station IDs; column order for every (S, n_sel) array",
        "station_node_index": "(6,) node index of each station in the full dataset",
        "horizons": "(4,) = [12,24,48,72] hours; column order for mae_*/rmse_*",
        "outage_rates_pct": "(4,) = [0,10,20,30]; axis-0 order for outage_gwn_72/outage_rf_72",
        "timestamps_h{H}": "(S,) datetime64 target time of each H-hour forecast",
        "true_h{H}": "(S, 6) ground-truth flow [veh/h] at horizon H",
        "gwn_h{H}": "(S, 6) GraphWaveNet-GRU-LSTM forecast [veh/h] at horizon H",
        "rf_h{H}": "(S, 6) Random Forest forecast [veh/h] at horizon H",
        "mae_gwn / rmse_gwn": "(6, 4) GraphWaveNet-GRU-LSTM error [veh/h], rows=stations cols=horizons",
        "mae_rf / rmse_rf": "(6, 4) Random Forest error [veh/h], rows=stations cols=horizons",
        "ev_timestamps": "(S,) datetime64 for the EV curves (EV_HORIZON grid)",
        "ev_load_actual_kw": "(S, 6) EV load from ground-truth flow [kW]",
        "ev_load_forecast_kw": "(S, 6) EV load from GraphWaveNet-GRU-LSTM forecast [kW]",
        "ev_flow_ref_max_actual": "(6,) per-station denominator max(actual flow) for ev_load_actual_kw [veh/h]",
        "ev_flow_ref_max_forecast": "(6,) per-station denominator max(forecast flow) for ev_load_forecast_kw [veh/h]",
        "outage_timestamps_72": "(S,) datetime64 for the 72h outage forecasts",
        "outage_true_72": "(S, 6) ground-truth flow [veh/h] at 72h",
        "outage_gwn_72": "(4, S, 6) GraphWaveNet-GRU-LSTM 72h forecast per outage rate [veh/h]",
        "outage_rf_72": "(4, S, 6) Random Forest 72h forecast per outage rate [veh/h]",
    },
    "ev_mapping_note": (
        "P_EV(t) = P_max * q(t) / max(q), with P_max=%.0f kW (thesis form). Each curve "
        "is normalised by its OWN per-station maximum on the EV_HORIZON=%dh grid: "
        "ev_load_actual_kw uses max(actual flow) (ev_flow_ref_max_actual) and "
        "ev_load_forecast_kw uses max(forecast flow) (ev_flow_ref_max_forecast), so "
        "both curves peak at %.0f kW." % (P_MAX_KW, EV_HORIZON, P_MAX_KW)
    ),
}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


with open(METADATA_JSON, "w") as f:
    json.dump(meta, f, indent=2, default=_json_default)
print("[export_bundle] Wrote %s" % METADATA_JSON)
print("[export_bundle] Done. Bundle is ready for the demo.")
