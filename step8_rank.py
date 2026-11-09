import pandas as pd
import numpy as np

df = pd.read_csv("screen_1BR5_exh8.csv")
df = df[df["e1"] != "ERR"].copy()
df["e1"] = pd.to_numeric(df["e1"])
df["heavy"] = pd.to_numeric(df["heavy"])
df["rotb"] = pd.to_numeric(df["rotb"])
df["pose1_dist_to_pocket"] = pd.to_numeric(df["pose1_dist_to_pocket"])

# controls
ctrl = df[df["id"].str.startswith("CTRL_")].set_index("id")
pta_le = ctrl.loc["CTRL_pteroic_acid","e1"] / ctrl.loc["CTRL_pteroic_acid","heavy"]
neo_le = ctrl.loc["CTRL_neopterin","e1"] / ctrl.loc["CTRL_neopterin","heavy"]
ctrl_le_floor = max(pta_le, neo_le)   # less negative = weaker = floor
print(f"Control ligand efficiencies:")
print(f"  Pteroic acid : {pta_le:.4f} kcal/mol/HA  (e1={ctrl.loc['CTRL_pteroic_acid','e1']:.2f})")
print(f"  Neopterin    : {neo_le:.4f} kcal/mol/HA  (e1={ctrl.loc['CTRL_neopterin','e1']:.2f})")
print(f"  Floor (weaker control): {ctrl_le_floor:.4f}")

drugs = df[~df["id"].str.startswith("CTRL_")].copy()
drugs["LE"] = drugs["e1"] / drugs["heavy"]

# filters
in_pocket  = drugs["pose1_dist_to_pocket"] <= 8.0
beats_ctrl = drugs["LE"] < ctrl_le_floor        # more negative = better
flagged    = drugs["rotb"] > 15

drugs["in_pocket"]  = in_pocket
drugs["beats_ctrl"] = beats_ctrl
drugs["flag_flex"]  = flagged

tier1 = drugs[in_pocket & beats_ctrl & ~flagged].sort_values("LE")
tier2 = drugs[in_pocket & beats_ctrl &  flagged].sort_values("LE")
tier3 = drugs[in_pocket & ~beats_ctrl].sort_values("LE")
outside = drugs[~in_pocket]

print(f"\nLibrary breakdown:")
print(f"  Total drugs docked:          {len(drugs)}")
print(f"  Pose outside pocket (>8 A):  {len(outside)}  -> excluded")
print(f"  In pocket, weaker than ctrl: {len(tier3)}  -> Tier 3 (low priority)")
print(f"  In pocket, beats ctrl, flex: {len(tier2)}  -> Tier 2 (flagged flexible)")
print(f"  In pocket, beats ctrl, rigid:{len(tier1)}  -> Tier 1 (shortlist)")

print(f"\n=== TIER 1 TOP 30 (by ligand efficiency, pose in pocket, LE better than both controls) ===")
cols = ["id","name","heavy","rotb","e1","LE","pose1_dist_to_pocket"]
print(tier1[cols].head(30).to_string(index=False))

print(f"\n=== TIER 2 TOP 10 (flexible, rotb>15) ===")
print(tier2[cols].head(10).to_string(index=False))

tier1.to_csv("results_tier1.csv", index=False)
tier2.to_csv("results_tier2.csv", index=False)
print("\n[OK] saved results_tier1.csv and results_tier2.csv")
