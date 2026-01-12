# **STU-NO**  
Learning PDE solution operators via an explicit, trainable SVD kernel layer  

---

## 🗺️ Repository layout
```text
STUNO/
├── STUNO/stu1d_mlp.py # 1-D core
├── STUNO/stu2d_mlp.py # 2-D core
├── STUNO/stu3d_mlp_vars.py # 3-D core
│
├── allen_cahn/
│ └── stu_ac_mlp.py # driver
└── … (other PDE folders)
```

### Core files
| File | Purpose |
|------|---------|
| `stu1d_mlp.py` | End-to-end 1-D operator (`STUNO1d`). |
| `stu2d_mlp.py / stu3d_mlp.py` | Same idea for 2-D / 3-D grids. |
---

## 📊  Paper results

| PDE                     | Driver script                     | Test 100 * rel. L₂ error (%) |
|-------------------------|-----------------------------------|-----------------------|
| 1-D Allen–Cahn          | `allen_cahn/stu_ac_mlp.py`         | **0.05** |
| 1-D Diffusion–Reaction  | `STUNO/dr/stu_dr_mlp.py`           | **0.22** |
| 1-D Diffusion–Sorption  | `STUNO/ds/stu_ds_mlp.py`           | **0.108**|
| 1-D Cahn-Hilliard       | `STUNO/cahn_hilliard/stu_ch_mlp.py`| **0.08** |
| 2-D Shallow Water       | `STUNO/sallow_water/stu_sw_mlp.py` | **0.38** |
| 3-D Maxwell.            | `STUNO/mx3/stu_mx3_mlp.py`         | **40.85**|

---

## 📥  Datasets

All benchmark data are publicly available.  
The first four can be downloaded from the **PDENNEval** collection on DaRUS  
<https://doi.org/10.18419/darus-2986>.

| PDE & Dim.               | Filename (DaRUS)                         | Size |
|--------------------------|------------------------------------------|------|
| 1-D Allen-Cahn           | `1D_Allen-Cahn_0.0001_5.hdf5`            | 3.9 GB |
| 1-D Diffusion–Reaction   | `ReacDiff_Nu0.5_Rho1.0.hdf5`             | 3.9 GB |
| 1-D Diffusion–Sorption   | `1D_diff-sorp_NA_NA.h5`                  | 4.0 GB |
| 2-D Shallow Water        | `2D_rdb_NA_NA.h5`                        | 6 GB |
---
and 1D Cahn-Hilliard and 3D Maxwell Equation can be downloaded from the links in: https://github.com/Sysuzqs/PDENNEval
