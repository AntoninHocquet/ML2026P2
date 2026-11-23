# ML-2026-P2: Clay Activation & R³ Pozzolanic Reactivity Database

This repository collects, harmonises and structures published experimental data on the pozzolanic reactivity of activated clays, measured by **R³ isothermal calorimetry** (Rapid, Relevant and Reliable; ASTM C1897-20 / RILEM TC 267-TRM). The goal is to predict R³ reactivity from the raw clay composition and the process settings, and to compare **calcination** with **mechanochemical activation (milling)** for a given clay.

---

## 📊 Data overview

| Dataset | File | Rows | Content |
| :--- | :--- | :---: | :--- |
| **Source dataset** | [`research/clay-activation-r3/r3_unified_dataset.csv`](research/clay-activation-r3/r3_unified_dataset.csv) | **268** | All curated samples with a 7-day R³ heat value: calcination (TA, 120) and milling (MCA, 148), with full traceability columns. Input of the build script |
| **Modelling datasets** | [`research/clay-activation-r3/processed/`](research/clay-activation-r3/processed/) | 111 + 142 | Cleaned calcination and milling tables (see next section) |
| **Original compilation** | [`research/clay-activation-r3/TA_MCA_MTA_R3_XRF_QXRD_20261111.xlsx`](research/clay-activation-r3/TA_MCA_MTA_R3_XRF_QXRD_20261111.xlsx) | 292 | TA and MCA sheets; also holds 28-day strength and strength activity index (122 rows), including 113 rows without an R³ value, 87 of which have a 28-day strength (useful for a future strength model) |
| **Source register** | [`research/clay-activation-r3/sources.json`](research/clay-activation-r3/sources.json) | 36 studies | Bibliographic metadata and DOIs (S01–S36), including studies not extracted yet and studies to avoid |
| **Literature report** | [`research/clay-activation-r3/Clay_Activation_R3_Research_Report_2026-11-16.pdf`](research/clay-activation-r3/Clay_Activation_R3_Research_Report_2026-11-16.pdf) | — | Evidence inventory and source assessment |

---

## 🎯 Final modelling datasets (calcination & milling)

Two ready-to-use tables, one per activation route. Each row is **one activated sample**: a raw clay treated under one set of process conditions, with its measured 7-day R³ heat.

| File | Rows | Distinct clays | Sources | Scope |
| :--- | :---: | :---: | :---: | :--- |
| [`research/clay-activation-r3/processed/calcination_dataset.csv`](research/clay-activation-r3/processed/calcination_dataset.csv) | 111 | 79 | 21 | Calcination with a dwell (static lab furnaces, one pilot rotary kiln), 400–1050 °C, air cooling. No flash calcination. |
| [`research/clay-activation-r3/processed/mechanical_dataset.csv`](research/clay-activation-r3/processed/mechanical_dataset.csv) | 142 | 28 | 17 | Laboratory planetary ball milling (Retsch PM100/PM400, Fritsch P6), clay milled alone. |

Rows removed while building the tables are listed with their reason in [`processed/dropped_rows.csv`](research/clay-activation-r3/processed/dropped_rows.csv) (duplicates, calcination without temperature, flash calcination, clay co-milled with glass / sodium silicate).

**Regenerate** (never edit the two CSVs by hand; fix `r3_unified_dataset.csv` or the script instead):

```
python3 scripts/build_datasets.py research/clay-activation-r3/r3_unified_dataset.csv research/clay-activation-r3/processed
python3 scripts/consistency.py research/clay-activation-r3/processed
```

### Columns

Missing counts are given as calcination / milling (out of 111 / 142 rows). "—" = column not in that file.

| Column | File(s) | Meaning (unit) | Role in a model | Missing | Missing-value handling |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `sample_id` | both | Unique sample identifier; join key to `r3_unified_dataset.csv` (reference URL, source table, notes, composition method, R³ quality label) | Identifier, **not a feature** | 0 / 0 | — |
| `parent_clay_id` | both | Raw clay identifier (same publication + same raw composition). The same clay appears in several rows (several temperatures or milling settings) and can appear in both files | **Group key for cross-validation** (`GroupKFold`), not a feature | 0 / 0 | — |
| `reference` | both | Source publication (author, year) | Filtering, leave-one-source-out validation; **not a feature** | 0 / 0 | — |
| `kaolinite_pct` | both | Kaolinite in the raw clay (wt%); the main reactive mineral once activated | Feature (composition) | 19 / 61 | 0 where the source mineralogy is complete (sum of all quantified phases ≥ 94 %) and for pure 2:1 reference minerals (illite, montmorillonite). Theoretical value (100) for the Plusquellec reference kaolinite. Otherwise left empty |
| `illite_mica_pct` | both | Illite + muscovite/mica (wt%); poorly reactive | Feature (composition) | 49 / 67 | 0 where the mineralogy is complete; otherwise left empty |
| `quartz_pct` | both | Quartz (wt%); inert, dilutes reactivity | Feature (composition) | 50 / 67 | 0 where the mineralogy is complete; otherwise left empty |
| `calcite_pct` | both | Calcite (wt%) | Feature (composition) | 55 / 91 | 0 where the mineralogy is complete; otherwise left empty |
| `sio2_pct` | both | SiO₂ of the raw clay, XRF (wt%) | Feature (composition) | 8 / 54 | Left empty (not measured ≠ 0) |
| `al2o3_pct` | both | Al₂O₃ (wt%); proxy for clay / kaolinite content | Feature (composition) | 8 / 54 | Left empty |
| `fe2o3_pct` | both | Fe₂O₃ (wt%); iron lowers reactivity | Feature (composition) | 9 / 59 | Left empty |
| `cao_pct` | both | CaO (wt%); mostly carbonates | Feature (composition) | 20 / 59 | Left empty |
| `mgo_pct` | both | MgO (wt%); smectite, dolomite | Feature (composition) | 20 / 61 | Left empty |
| `k2o_pct` | both | K₂O (wt%); proxy for illite / mica | Feature (composition) | 18 / 59 | Left empty |
| `loi_pct` | both | Loss on ignition (wt%): structural water (kaolinite ≈ 14 %) + CO₂ of carbonates | Feature (composition) | 33 / 72 | Left empty |
| `calcination_temp_c` | calcination | Calcination temperature (°C) | Feature (process); the most important one. Non-monotonic effect (over-calcination above ~900 °C) | 0 / — | Rows without temperature are removed |
| `dwell_min` | calcination | Time at temperature (min) | Feature (process) | 10 / — | Left empty in the file. At model time: native NaN handling, or training-fold median (60 min) + a `dwell_missing` indicator. Never 0 |
| `milling_time_min` | milling | Milling duration (min) | Feature (process) | — / 0 | — |
| `mill_rpm` | milling | Mill rotation speed (rpm); the most important milling feature | Feature (process) | — / 5 | Left empty |
| `ball_to_powder_ratio` | milling | Ball-to-powder mass ratio (–) | Feature (process) | — / 11 | Left empty |
| `grinding_aid_tea_wt_pct` | milling | Triethanolamine grinding aid (wt%) | Feature (process) | — / 0 | 0 when no grinding aid is reported (Plusquellec and Muzenda used TEA) |
| `r3_heat_7d_j_per_g` | both | **Target**: cumulative R³ heat at 7 days, 40 °C (J per g of activated clay), ASTM C1897 / RILEM / EN 17979 method A | **Target** | 0 / 0 | — |

### Missing values: rules

- **A 0 is written only when the absence is known**: a mineral not detected in a complete mineralogy, kaolinite in a pure 2:1 mineral, no grinding aid. Oxides and process settings are never set to 0.
- **Everything else is left empty** (= not measured). Recommended model: `HistGradientBoostingRegressor`, which handles missing values natively. For models that cannot (random forest, linear models), impute with the **median of the training fold only**, inside the cross-validation, and add a `*_missing` indicator column. Do not impute with statistics computed on the whole table (leakage), and do not use a global mean for minerals (missing values come from whole sources or clay types).
- **Known gap**: 48 milling rows (Marsh 2026 clays "K", "Mt" and "F") have no composition at all; it is not published (paper or Zenodo).

### How to validate

Always split by `parent_clay_id` (e.g. `GroupKFold`), so that the same clay never appears in both training and test. Baselines (RMSE of predicting the mean): 222 J/g for calcination, 159 J/g for milling.

---

## 🔬 Activation routes covered

1. **Calcination (thermal activation, TA)**: static laboratory furnaces and one pilot rotary kiln, 400–1050 °C, dwell 15–240 min, air cooling. Flash calcination and reducing-atmosphere cooling were excluded (not comparable).
2. **Milling (mechanochemical activation, MCA)**: laboratory planetary ball mills (Retsch PM100/PM400, Fritsch P6), 60–650 rpm, 5–720 min, ball-to-powder ratio 2–35, optional triethanolamine grinding aid.

*Removed from the repository*: combined mechano-thermal activation (MTA: 17 R³ values on about 6 clays, too few for a model) and the Bahhou 2026 secondary compilation (125 rows without per-row primary sources; adding it degraded the grouped cross-validation error from 161 to 188 J/g).

---

## 📋 Source dataset dictionary (`r3_unified_dataset.csv`)

- **Identifiers & target**: `Sample_ID (unified)`, `Activation_type` (TA, MCA), `R3_7d (J/g SCM)`, `Original sample ID`
- **Raw clay quantitative mineralogy (wt%)**: `Kaolinite`, `Smectite / Mnt`, `Illite / Mica`, `Mixed I/S`, `Chlorite`, `Quartz`, `Calcite`, `Dolomite`, `Feldspar`, `Other minerals`, `Raw amorphous`, `Total clay`
- **Raw clay chemistry, XRF (wt% oxides)**: `Al2O3`, `SiO2`, `Fe2O3`, `CaO`, `MgO`, `K2O`, `Na2O`, `TiO2`, `LOI`
- **Process settings**: `TA temperature (°C)`, `TA hold (min)`, `Milling time (min)`, `Milling energy (kWh/kg SCM)`, `D50 (µm)`
- **Traceability & quality**: `Reference (author-year)`, `Reference URL`, `Source table / figure`, `R3 data quality`, `Composition source / method`, `Notes / QC`, `R3 protocol / comparability`

Milling speed, ball-to-powder ratio and grinding aid are not separate columns in the source: `build_datasets.py` extracts them from the sample IDs and the notes.
