# FaceMoCap Alignment & Mean Movement

This repository provides utilities to (1) **remove head motion** from FaceMoCap marker trajectories using a **dental coordinate frame**, (2) **rigidly align** each sequence to a **template neutral face**, (3) extract the **active movement window**, resample to a fixed length, and compute **per-movement healthy mean trajectories**.  
It also includes a script to score a single sample against the healthy mean (kinesia/anomaly direction metrics).

---

## Repository map (folders / modules)

```text
facemocap-alignment-main/
├── align_and_mean_movement.py
├── anomaly_kinesia_direction_single_movement_v3.py
├── scripts/
│   └── align_and_mean_movement_v1.py
└── align_mean_movement/
    ├── __init__.py
    ├── cli.py
    ├── pipeline.py
    ├── geometry/
    │   ├── __init__.py
    │   ├── dental_frame.py
    │   ├── rigid_alignment.py
    │   └── rotations.py
    ├── io/
    │   ├── __init__.py
    │   ├── facemocap_csv.py
    │   ├── metadata.py
    │   └── outputs.py
    ├── processing/
    │   ├── __init__.py
    │   ├── movement_window.py
    │   ├── neutral.py
    │   └── resampling.py
    └── viz/
        ├── __init__.py
        └── plotly_anim.py
```

---

## Functionality map (what is where)

> This is an “index” so a new user can quickly find the implementation.

### High-level entry points
- **`align_and_mean_movement.py`** — wrapper entry point for the full pipeline.
- **`scripts/align_and_mean_movement_v1.py`** — legacy wrapper (same pipeline; older filename).
- **`align_mean_movement/cli.py`** — main CLI orchestration (`main()`).
- **`anomaly_kinesia_direction_single_movement_v3.py`** — single-sample scoring vs mean (`main()`).

### Package-level modules

**`align_mean_movement/cli.py`**
- main()  # CLI entry point for alignment + mean computation

**`align_mean_movement/pipeline.py`**
- (reserved placeholder)

**`align_mean_movement/io/facemocap_csv.py`**
- read_facemocap_csv()
- reshape_markers()
- split_dental_face()

**`align_mean_movement/io/metadata.py`**
- infer_columns()
- normalize_movement_label()
- iter_samples_from_metadata()

**`align_mean_movement/io/outputs.py`**
- ensure_dir()
- write_csv()
- write_npy()
- write_json()

**`align_mean_movement/geometry/dental_frame.py`**
- compute_dental_frame()
- transform_to_dental_frame()
- apply_fixed_rotation_xyz()

**`align_mean_movement/geometry/rotations.py`**
- rotation_matrix_from_axis_angle()
- euler_xyz_deg_to_R()

**`align_mean_movement/geometry/rigid_alignment.py`**
- nanmean_keepdims()
- nanmedian()
- rms()
- trimmed_indices()
- kabsch()
- apply_rt()
- trimmed_ransac_rigid()
- huber_irls_refine()
- maybe_yaw_flip()

**`align_mean_movement/processing/neutral.py`**
- select_neutral_frame()

**`align_mean_movement/processing/movement_window.py`**
- movement_energy()
- extract_active_window()
- fill_small_gaps()

**`align_mean_movement/processing/resampling.py`**
- resample_nan_robust()
- linear_interp_nan_robust()

**`align_mean_movement/viz/plotly_anim.py`**
- make_animation_html()
- scatter3d_frames()

**`align_and_mean_movement.py`**
- main()  # wrapper (delegates to align_mean_movement.cli)

**`scripts/align_and_mean_movement_v1.py`**
- main()  # legacy wrapper (delegates to align_mean_movement.cli)

**`anomaly_kinesia_direction_single_movement_v3.py`**
- main()  # single-sample scoring vs healthy mean (kinesia/anomaly)


---

## Installation (Windows / macOS / Linux)

This repo is meant to be run **from the repository root**. It is not shipped as a pip package (no `pyproject.toml`/`setup.py`), so you typically just install dependencies and run the scripts.

### Requirements
- Python **3.9+** recommended
- Dependencies: `numpy`, `pandas`, `plotly`, `matplotlib`

---

### Option A — Conda (recommended)

```bash
conda create -n facemocap_alignment python=3.11 -y
conda activate facemocap_alignment

pip install -U pip
pip install numpy pandas plotly matplotlib
```

---

### Option B — venv

#### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install numpy pandas plotly matplotlib
```

#### Windows (PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -U pip
pip install numpy pandas plotly matplotlib
```

---

## Input assumptions

### FaceMoCap CSV layout
The default configuration assumes:
- numeric data begins after **5 header rows** (`skiprows=5`)
- columns `2..325` are numeric XYZ (end exclusive 326)  
  → 324 columns → **108 points × 3**
- the first **3 points** are **dental markers** (used to define the dental frame)

If your dataset differs, adjust the anomaly script flags (`--skiprows`, `--usecols_start`, `--usecols_end`, `--dental_n`) and/or edit `align_mean_movement/io/facemocap_csv.py`.

### Metadata CSV
The pipeline expects a metadata CSV that contains:
- a column with the file path to the CSV sample (absolute or relative)
- a movement label (e.g., `M1..M5`)
- a group/condition label to identify healthy samples

The **anomaly script** requires a column named **`complete_filepath`** (exact match is used).

---

## Usage

## 1) Align sequences + compute healthy mean per movement

Wrapper entry points:
- `python align_and_mean_movement.py ...`
- `python scripts/align_and_mean_movement_v1.py ...`

Example:
```bash
python align_and_mean_movement.py   --metadata /path/to/facemocap_metadata.csv   --template_csv /path/to/template.csv   --out_dir /path/to/output_dir   --root_override "/path/to/Data_FaceMoCap"   --movements M1 M2 M3 M4 M5   --n_frames 100   --neutral_first_pct 0.05   --energy_thr_percentile 70   --min_window_len 10   --min_points 60   --ransac   --ransac_trials 4000   --ransac_subset 4   --trim_frac 0.10   --try_yaw_flip   --yaw_flip_axis Z   --huber_iters 3   --huber_k 1.5   --fixed_rot_xyz 90 90 90   --export_sample_anims 3   --seed 0
```

### Alignment CLI flags (complete list)
{format_args(cli_args)}

### Outputs (typical)
In `--out_dir`:
- `summary_sequences.csv` — per-sample diagnostics and status
- `template_neutral_frame0.html` — Plotly HTML view of template neutral
- Per movement `Mk/`:
  - `mean_healthy.npy`
  - `mean_healthy.csv`
  - `mean_healthy_animation.html`
  - a few `sample_animation_*.html` (controlled by `--export_sample_anims`)

---

## 2) Score a single sample vs the healthy mean (kinesia/anomaly)

Prerequisite: you already ran step (1) so you have `Mk/mean_healthy.npy`.

Example:
```bash
python anomaly_kinesia_direction_single_movement_v3.py   --metadata /path/to/facemocap_metadata.csv   --target_complete_filepath "/absolute/or/metadata/value.csv"   --root_override "/path/to/Data_FaceMoCap"   --ref_root /path/to/output_dir   --out_dir /path/to/anomaly_outputs   --n_frames 100   --neutral_first_pct 0.05   --energy_thr_percentile 40   --min_window_len 30   --max_gap 5   --trim_frac 0.10   --huber_iters 3   --huber_k 1.5   --try_yaw_flip   --yaw_flip_axis Z   --fixed_rot_xyz 90 90 90   --time_align lag   --lag_max 10   --top_k 10   --topk_mode twosided
```

### Anomaly script CLI flags (complete list)
{format_args(anom_args)}

### `--ref_root` expected structure
```text
ref_root/
  M1/mean_healthy.npy
  M2/mean_healthy.npy
  ...
```

---

## Troubleshooting

- **Plotly errors / missing HTML outputs** → `pip install plotly`
- **Many failures due to dental frame** → check that the first 3 markers are valid in most frames.
- **Low inlier count / bad alignment** → increase `--min_points`, tune `--trim_frac`, enable `--try_yaw_flip`, or increase `--ransac_trials`.
- **Window extraction too short / wrong** → tune `--energy_thr_percentile` and `--min_window_len`.

---

## Notes for packaging (optional)

If you want `pip install -e .`, add a minimal `pyproject.toml` and include `align_mean_movement*` in the package discovery. This repository currently runs fine without packaging (run from repo root).

