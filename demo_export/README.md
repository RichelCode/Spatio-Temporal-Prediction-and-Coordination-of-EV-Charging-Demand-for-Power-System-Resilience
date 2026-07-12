# demo_export — small demo bundle

Standalone exporter that packages a **few-MB demo** of the thesis pipeline for
~6 representative PeMS District 3 stations (station **308511** is always
included). It compares the **GraphWaveNet-GRU-LSTM** model against the
**Random Forest** baseline, applies the traffic→EV mapping, and runs the
sensor-outage robustness check.

It only **reads** your trained weights + dataset; it never modifies the model or
training notebooks and writes only inside `demo_export/`.

## How to run (on Paperspace)

1. Open `export_bundle.py` and set the two paths near the top:

   ```python
   CHECKPOINT_PATH = "artifacts/runs/<...>_GraphWaveNet_GRU_LSTM/best.pt"  # or the run folder
   DATA_PATH       = "artifacts/pems_graph_dataset_strict.npz"
   ```

   `CHECKPOINT_PATH` accepts either the `best.pt` file or the run directory that
   contains it. Wrong paths produce a friendly error telling you what to fix.

2. In the same environment you trained in (needs `torch`, `scikit-learn`,
   `pandas`, `numpy`):

   ```bash
   python demo_export/export_bundle.py
   ```

3. Outputs (written next to the script):
   - `demo_export/bundle.npz` — all numeric arrays
   - `demo_export/metadata.json` — human-readable description, config, and metrics

The Random Forest is refit from scratch **only for the selected stations** (it is
per-station, so this is cheap). The graph model runs one clean pass plus one pass
per non-zero outage rate over the test set.

## Loading the bundle

```python
import numpy as np, json
b = np.load("demo_export/bundle.npz", allow_pickle=True)
meta = json.load(open("demo_export/metadata.json"))

stations = b["stations"]          # e.g. ['308511', ...]  -> column order of every (S, 6) array
horizons = b["horizons"]          # [12, 24, 48, 72]      -> column order of mae_*/rmse_*
s = list(stations).index("308511")

t   = b["timestamps_h72"]         # (S,) datetime64 target times of the 72h forecast
yt  = b["true_h72"][:, s]         # ground-truth flow  [veh/h]
yg  = b["gwn_h72"][:, s]          # GraphWaveNet-GRU-LSTM forecast [veh/h]
yr  = b["rf_h72"][:, s]           # Random Forest forecast [veh/h]
```

## Conventions

- **S** = number of test windows (one forecast origin per hour). Row `i` of every
  per-horizon array is one forecast; its target time is `timestamps_h{H}[i]`.
- **n_sel = 6** stations. The column order in every `(S, 6)` and `(6, …)` array is
  exactly `bundle["stations"]`.
- Forecasts are **de-scaled** to real units (the model trains on z-scored flow).
- **Units:** flow = **vehicles/hour**, power = **kW**, timestamps =
  `np.datetime64[ns]` (hourly).

## Arrays in `bundle.npz`

| Array | Shape | Units | Meaning |
|---|---|---|---|
| `stations` | (6,) | — | Station IDs; column order for all per-station arrays |
| `station_node_index` | (6,) | — | Each station's node index in the full dataset |
| `horizons` | (4,) | hours | `[12, 24, 48, 72]`; column order for `mae_*`/`rmse_*` |
| `outage_rates_pct` | (4,) | % | `[0, 10, 20, 30]`; axis-0 order for the outage arrays |
| `timestamps_h{H}` | (S,) | datetime64 | Target time of each H-hour forecast (H ∈ 12/24/48/72) |
| `true_h{H}` | (S, 6) | veh/h | Ground-truth flow at horizon H |
| `gwn_h{H}` | (S, 6) | veh/h | GraphWaveNet-GRU-LSTM forecast at horizon H |
| `rf_h{H}` | (S, 6) | veh/h | Random Forest forecast at horizon H |
| `mae_gwn`, `rmse_gwn` | (6, 4) | veh/h | Graph model error; rows = `stations`, cols = `horizons` |
| `mae_rf`, `rmse_rf` | (6, 4) | veh/h | Random Forest error; rows = `stations`, cols = `horizons` |
| `ev_timestamps` | (S,) | datetime64 | Time grid for the EV curves (the `EV_HORIZON`=24h grid) |
| `ev_load_actual_kw` | (S, 6) | kW | EV load from **ground-truth** flow |
| `ev_load_forecast_kw` | (S, 6) | kW | EV load from the **graph-model forecast** |
| `ev_flow_ref_max_actual` | (6,) | veh/h | Per-station denominator `max(actual flow)` for `ev_load_actual_kw` |
| `ev_flow_ref_max_forecast` | (6,) | veh/h | Per-station denominator `max(forecast flow)` for `ev_load_forecast_kw` |
| `ev_horizon` | scalar | hours | Horizon used for the EV curves (24) |
| `p_max_kw` | scalar | kW | `P_max` (200) |
| `outage_timestamps_72` | (S,) | datetime64 | Target time of each 72h outage forecast |
| `outage_true_72` | (S, 6) | veh/h | Ground-truth flow at 72h |
| `outage_gwn_72` | (4, S, 6) | veh/h | Graph model 72h forecast per outage rate (axis 0 = 0/10/20/30%) |
| `outage_rf_72` | (4, S, 6) | veh/h | Random Forest 72h forecast per outage rate |

## EV mapping

```
P_EV(t) = P_max · q(t) / max(q)        P_max = 200 kW
```

Literal per-series form from the thesis: **each curve is normalised by its own
maximum**, so both `ev_load_actual_kw` and `ev_load_forecast_kw` peak at 200 kW.
`ev_load_actual_kw` uses `max(actual flow)` (`ev_flow_ref_max_actual`) and
`ev_load_forecast_kw` uses `max(forecast flow)` (`ev_flow_ref_max_forecast`), both
per station on the 24h grid.

## Sensor-outage forecasts

For each rate in `{0, 10, 20, 30}%`, a fixed fraction of network sensor nodes is
zeroed at the model input (fixed seed `OUTAGE_SEED=0`, reproducible). `0%` is the
clean forecast. The graph model sees degraded network context; the Random Forest
(which is per-station and has no cross-node inputs) is affected only in windows
where the station itself is out. Compare against `outage_true_72`.

## metadata.json

Mirrors the config and adds `metrics_per_station_per_horizon` (MAE/RMSE for both
models) and `outage_mae72_per_station_per_rate`, plus the inferred model
architecture and the exact checkpoint/dataset paths used.
