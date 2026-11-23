"""Consistency checks on calcination_dataset.csv + mechanical_dataset.csv.
Usage: python3 consistency.py <processed_dir>"""
import sys, numpy as np, pandas as pd

a = pd.concat([pd.read_csv(f"{sys.argv[1]}/{f}_dataset.csv", low_memory=False).assign(process_family=fam)
               for f, fam in (("calcination", "thermal"), ("mechanical", "mechanical"))], ignore_index=True)
y = "r3_heat_7d_j_per_g"
print("rows:", len(a))
print("\n[1] ranges by family"); print(a.groupby("process_family")[y].describe().round(0).to_string())
print("\n[2] > 1000 J/g"); print(a[a[y] > 1000][["reference", "sample_id", y, "calcination_temp_c"]].to_string())
t = a[(a.process_family == "thermal") & a.kaolinite_pct.notna() & a.calcination_temp_c.between(650, 850)]
if len(t) > 5:
    s, i = np.polyfit(t.kaolinite_pct, t[y], 1); r = np.corrcoef(t.kaolinite_pct, t[y])[0, 1]
    print(f"\n[3] thermal 650-850 C, n={len(t)}: heat = {s:.1f}*kaolinite + {i:.0f}, r={r:.2f}")
key = ["parent_clay_id", "process_family", "calcination_temp_c", "dwell_min",
       "milling_time_min", "mill_rpm", "ball_to_powder_ratio", "grinding_aid_tea_wt_pct"]
d = a[a.duplicated(key, keep=False)]
print(f"\n[5] same clay + same recorded conditions, several rows: {len(d)}")
print(d.sort_values("parent_clay_id")[["parent_clay_id", "sample_id", y]].head(30).to_string())
