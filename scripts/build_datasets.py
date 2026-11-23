"""Build the calcination / milling modelling datasets from r3_unified_dataset.csv.

r3_unified_dataset.csv (TA + MCA) is the only source: it replaces the Excel file
TA_MCA_MTA and the former r3_measurements / r3_auxiliary_measurements (deleted).

Usage: python3 build_datasets.py <r3_unified_dataset.csv> <out_dir>
"""
import sys, re, unicodedata
import numpy as np
import pandas as pd

src, out = sys.argv[1:3]
u = pd.read_csv(src)
n_in = len(u)

RENAME = {
    "Sample_ID (unified)": "sample_id", "Activation_type": "activation_type",
    "R3_7d (J/g SCM)": "r3_heat_7d_j_per_g", "Original sample ID": "original_sample_id",
    "Kaolinite (wt%)": "kaolinite_pct", "Smectite / Mnt (wt%)": "smectite_pct",
    "Illite / Mica (wt%)": "illite_mica_pct", "Mixed I/S (wt%)": "mixed_is_pct",
    "Chlorite (wt%)": "chlorite_pct", "Quartz (wt%)": "quartz_pct", "Calcite (wt%)": "calcite_pct",
    "Dolomite (wt%)": "dolomite_pct", "Feldspar (wt%)": "feldspar_pct",
    "Other minerals (wt%)": "other_minerals_pct", "Raw amorphous (wt%)": "raw_amorphous_pct",
    "Total clay (reported wt%)": "total_clay_pct", "Al2O3 (wt%)": "al2o3_pct",
    "SiO2 (wt%)": "sio2_pct", "Fe2O3 (wt%)": "fe2o3_pct", "CaO (wt%)": "cao_pct",
    "MgO (wt%)": "mgo_pct", "K2O (wt%)": "k2o_pct", "Na2O (wt%)": "na2o_pct",
    "TiO2 (wt%)": "tio2_pct", "LOI (wt%)": "loi_pct",
    "TA temperature (°C)": "calcination_temp_c", "TA hold (min)": "dwell_min",
    "Milling time (min)": "milling_time_min", "Milling energy (kWh/kg SCM)": "milling_energy_kwh_per_kg",
    "D50 (µm)": "post_d50_um", "Reference (author-year)": "reference", "Reference URL": "reference_url",
    "Source table / figure": "source_locator", "R3 data quality": "r3_data_quality_raw",
    "Composition source / method": "composition_method", "Notes / QC": "notes",
    "R3 protocol / comparability": "r3_protocol",
}
missing = set(RENAME) - set(u.columns)
assert not missing, missing
d = u.rename(columns=RENAME)
d["r3_heat_7d_j_per_g"] = pd.to_numeric(d.r3_heat_7d_j_per_g, errors="coerce")
assert d.r3_heat_7d_j_per_g.notna().all()

TXT = (d.notes.fillna("") + " | " + d.source_locator.fillna("") + " | " + d.r3_protocol.fillna("")).str.lower()
SID = d.sample_id.astype(str)

# ------------------------------------------------------------------ 1. exclusions
log = []
def drop(mask, reason):
    global d, TXT, SID
    for _, r in d[mask].iterrows():
        log.append({"sample_id": r.sample_id, "reference": r.reference,
                    "r3_heat_7d_j_per_g": r.r3_heat_7d_j_per_g, "reason": reason})
    keep = ~mask
    d, TXT, SID = d[keep].copy(), TXT[keep], SID[keep]

# Vallina 2024 (S07) = same samples as Koufany 2025 Ref1/Ref2 (same group, same T; exact values in Koufany)
drop(SID.isin(["Vallina24_SAu-1_T750", "Vallina24_SWy-3_T800"]),
     "duplicate of Koufany25_Ref1-SCC / Ref2-SCC (exact value kept)")

d["process_family"] = d.activation_type.map({"TA": "thermal", "MCA": "mechanical"})
drop((d.process_family == "thermal") & d.calcination_temp_c.isna(),
     "calcination without temperature (Martirena 2026: oxidising / reducing cooling, T not reported)")
drop(SID == "Panzer24_PP", "flash calcination (seconds): temperature not comparable to a furnace dwell")

# ------------------------------------------------------------------ 2. variables extracted from notes / IDs
def first_num(pattern, s):
    m = re.search(pattern, s)
    return float(m.group(1)) if m else np.nan

rpm, bpr, aid = [], [], []
for sid, t, fam in zip(SID, TXT, d.process_family):
    if fam != "mechanical":
        rpm.append(np.nan); bpr.append(np.nan); aid.append(np.nan); continue
    # speed: sample code first (row-specific), otherwise notes
    r = np.nan
    for pat in [r"MuscMCA_(\d{3})_", r"DoE_(\d{3})_", r"_M(\d{3})_\d+", r"_M(\d{3})(?:_|$)",
                r"_H(\d{3})_", r"KM-(\d{3})-"]:
        m = re.search(pat, sid)
        if m:
            r = float(m.group(1)); break
    if np.isnan(r):
        r = first_num(r"(\d{2,4})\s*rpm", t)
    rpm.append(r)
    b = first_num(r"BPR(\d+)", sid)
    if np.isnan(b):
        b = first_num(r"bpr\s*(?:of\s*)?~?\s*(\d+(?:\.\d+)?)", t)
    if np.isnan(b):
        b = first_num(r"(\d+(?:\.\d+)?)\s*:\s*1\s*bpr", t)
    bpr.append(b)
    a = re.search(r"TEA(\d)p(\d+)", sid)
    if a:
        aid.append(float(f"{a.group(1)}.{a.group(2)}"))
    else:
        aid.append(first_num(r"(\d+(?:\.\d+)?)\s*wt%\s*(?:triethanolamine|tea)", t))
d["mill_rpm"], d["ball_to_powder_ratio"], d["grinding_aid_tea_wt_pct"] = rpm, bpr, aid


# co-milling with an additive (Callender: waste glass powder WGP, sodium silicate), wt%
def pct(pat, sid):
    m = re.search(pat, sid)
    return float(m.group(1).replace("_", ".")) if m else 0.0
d["co_milled_glass_pct"] = [pct(r"_(\d+(?:_\d)?)WGP", sid) if f == "mechanical" else np.nan for sid, f in zip(SID, d.process_family)]
d["co_milled_na2sio3_pct"] = [pct(r"_(\d+(?:_\d)?)Na2SiO3", sid) if f == "mechanical" else np.nan for sid, f in zip(SID, d.process_family)]
# co-milling: the R3 includes the reactivity of the glass / silicate (another material) -> rows removed
drop((d.co_milled_glass_pct > 0) | (d.co_milled_na2sio3_pct > 0),
     "co-milled with glass / sodium silicate (Callender): R3 of another material")

# ------------------------------------------------------------------ 3. parent_clay_id
COMP = ["sio2_pct", "al2o3_pct", "fe2o3_pct", "kaolinite_pct", "smectite_pct", "illite_mica_pct"]
def ascii_surname(s):
    head = re.match(r"^([^\s,(]+)", str(s)).group(1)
    norm = unicodedata.normalize("NFKD", head).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z]", "", norm)
ref_key = d.reference.map(ascii_surname)
ckey = d[COMP].apply(lambda r: "" if r.isna().all() else "|".join("na" if pd.isna(v) else f"{v:.1f}" for v in r), axis=1)
codes = {k: i for i, k in enumerate(sorted(set(zip(ref_key, ckey))), 1)}
# no composition: clay = ID prefix before the treatment code (e.g. Marsh26_K_D5 -> Marsh26_K)
def stem(sid):
    parts = sid.split("_")
    return "_".join(parts[:2]) if len(parts) > 2 else sid
d["parent_clay_id"] = [f"{rk}-C{codes[(rk, ck)]:03d}" if ck else f"{rk}-{stem(s)}"
                       for rk, ck, s in zip(ref_key, ckey, SID)]
# Alvarez-Coscojuela 2025 and Mañosa 2024/2025 use the exact same commercial kaolin
# (identical XRF: SiO2=49.85, Al2O3=36.31, Fe2O3=0.47, LOI=12.03) -> merge to prevent CV leakage
manosa_kln_id = d.loc[SID == "Manosa25_LC3-MK", "parent_clay_id"].iloc[0]
d.loc[SID.str.startswith("Alvarez25_"), "parent_clay_id"] = manosa_kln_id

# exact internal duplicates (same clay, same conditions, same value)
key = ["parent_clay_id", "process_family", "calcination_temp_c", "dwell_min", "milling_time_min",
       "mill_rpm", "ball_to_powder_ratio", "grinding_aid_tea_wt_pct", "r3_heat_7d_j_per_g"]
dup = d.duplicated(key, keep="first") & d[COMP].notna().any(axis=1)
drop(dup, "exact internal duplicate (same clay, same conditions, same value)")

# ------------------------------------------------------------------ 4. known zeros
# complete mineralogy (sum of all quantified phases >= 94 %): an unlisted phase was not
# detected -> 0. The 94 threshold includes Mañosa (94.8 %: 5.1 % amorphous excluded by the authors).
# Exclude nominal mixtures that are not XRD measurements (e.g. Plusquellec23_IllQ80_20_M300_60).
ALL_MIN = ["kaolinite_pct", "smectite_pct", "illite_mica_pct", "mixed_is_pct", "chlorite_pct", "quartz_pct",
           "calcite_pct", "dolomite_pct", "feldspar_pct", "other_minerals_pct", "raw_amorphous_pct"]
OUT_MIN = ["kaolinite_pct", "illite_mica_pct", "quartz_pct", "calcite_pct"]
complete = (d[ALL_MIN].sum(axis=1, min_count=1) >= 94) & ~TXT.str.contains("not xrd")
for c in OUT_MIN:
    d.loc[complete, c] = d.loc[complete, c].fillna(0)
# pure 2:1 reference minerals (illite, montmorillonite): no kaolinite by definition
pure_2to1 = SID.str.contains(r"^(?:Zhao25_(?:Ilt|Mnt)|Marsh24Mnt_|Marsh23ICCC_Mnt|Plusquellec23_Ill_|Koufany25_Ref[12]-SCC)")
d.loc[pure_2to1, "kaolinite_pct"] = d.loc[pure_2to1, "kaolinite_pct"].fillna(0)
# grinding aid: not reported = not used
mech = d.process_family == "mechanical"
d.loc[mech, "grinding_aid_tea_wt_pct"] = d.loc[mech, "grinding_aid_tea_wt_pct"].fillna(0)
print(f"known zeros: complete mineralogy {int(complete.sum())} rows, pure 2:1 {int(pure_2to1.sum())} rows")

# ------------------------------------------------------------------ 5. output
# sample_id: key (join with r3_unified_dataset for traceability); parent_clay_id: CV groups;
# reference: filtering / leave-one-source-out CV
ID = ["sample_id", "parent_clay_id", "reference"]
# feldspar, other_minerals, raw_amorphous, total_clay, smectite, mixed_is, chlorite, dolomite:
# minor phases / few clays, no gain in grouped CV -> not exported
# (kept in r3_unified_dataset.csv; needed to judge whether a mineralogy is complete)
CLAY = ["kaolinite_pct", "illite_mica_pct", "quartz_pct", "calcite_pct",
        "sio2_pct", "al2o3_pct", "fe2o3_pct", "cao_pct", "mgo_pct", "k2o_pct",
        "loi_pct"]  # na2o, tio2: no gain in grouped CV
THERMAL = ["calcination_temp_c", "dwell_min"]
MECH = ["milling_time_min", "mill_rpm", "ball_to_powder_ratio", "grinding_aid_tea_wt_pct"] # milling_energy: 3 clays (Marsh 2026), worse RMSE
POST = []  # post_d50_um: no gain; in milling it is an outcome of milling (leakage)
TARGET = ["r3_heat_7d_j_per_g"]
# composition_method and notes stay in r3_unified_dataset.csv (join on sample_id)

allx = d[ID + ["process_family"] + CLAY + THERMAL + MECH + POST + TARGET].reset_index(drop=True)
th = allx[allx.process_family == "thermal"][ID + CLAY + THERMAL + POST + TARGET]
me = allx[allx.process_family == "mechanical"][ID + CLAY + MECH + POST + TARGET]
th.to_csv(f"{out}/calcination_dataset.csv", index=False)
me.to_csv(f"{out}/mechanical_dataset.csv", index=False)
pd.DataFrame(log).to_csv(f"{out}/dropped_rows.csv", index=False)

# ------------------------------------------------------------------ 6. checks
assert len(th) + len(me) == len(allx) == n_in - len(log)
assert allx.sample_id.is_unique
print(f"input {n_in} | dropped {len(log)} | thermal {len(th)} | mechanical {len(me)}")
print(pd.DataFrame(log).reason.value_counts().to_string())
for name, df in [("thermal", th), ("mechanical", me)]:
    print(f"\n== {name}: rows {len(df)}, parent clays {df.parent_clay_id.nunique()}")
    cols = THERMAL if name == "thermal" else MECH
    print(" filled:", {c: int(df[c].notna().sum()) for c in cols})
pt = set(th.parent_clay_id); pm = set(me.parent_clay_id)
print("\nclays with both thermal and mechanical:", len(pt & pm))
