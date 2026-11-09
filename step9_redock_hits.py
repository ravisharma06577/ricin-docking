import os, csv
import numpy as np
import pandas as pd
from vina import Vina

# top 50 by LE from tier1, but manually prioritise mechanistically interesting ones
tier1 = pd.read_csv("results_tier1.csv").sort_values("LE").head(50)
ids = tier1["id"].tolist()
# ensure allopurinol and isoniazid are included
for must in ["CHEMBL1467", "CHEMBL64"]:
    if must not in ids: ids.append(must)

print(f"Re-docking {len(ids)} hits at exhaustiveness 64 in both receptors")
os.makedirs("redock", exist_ok=True)
out = "results_redock.csv"

done = set()
if os.path.exists(out):
    done = set(pd.read_csv(out)["id"] + pd.read_csv(out)["receptor"])

f = open(out, "a", newline=""); w = csv.writer(f)
if not os.path.exists(out) or os.path.getsize(out) == 0:
    w.writerow(["id","name","receptor","e1","e2","e3","pose1_dist","seconds"])
    f.flush()

import time
pocket = {}
for rid in ("1BR5","1BR6"):
    xyz = [[float(l[30:38]),float(l[38:46]),float(l[46:54])]
           for l in open(f"protein/{rid}_ligand.pdb") if l.startswith("HETATM")]
    pocket[rid] = np.mean(xyz, axis=0)

for rid in ("1BR5","1BR6"):
    rec  = f"protein/{rid}_rec.pdbqt"
    box  = np.loadtxt(f"protein/{rid}_box.txt")
    v = Vina(sf_name="vina", cpu=0, seed=42, verbosity=0)
    v.set_receptor(rec)
    v.compute_vina_maps(center=box[0].tolist(), box_size=[16.0]*3)
    for cid in ids:
        key = cid + rid
        if key in done: continue
        row = tier1[tier1.id == cid]
        name = row["name"].values[0] if len(row) else cid
        t = time.time()
        try:
            v.set_ligand_from_file(f"library/pdbqt/{cid}.pdbqt")
            v.dock(exhaustiveness=64, n_poses=5)
            E = v.energies(n_poses=5)
            scores = [round(float(e[0]),3) for e in E] + [""]*5
            p = v.poses(n_poses=1)
            c = np.mean([[float(l[30:38]),float(l[38:46]),float(l[46:54])]
                         for l in p.splitlines() if l.startswith(("ATOM","HETATM"))],axis=0)
            dist = round(float(np.linalg.norm(c - pocket[rid])),2)
            v.write_poses(f"redock/{cid}_{rid}.pdbqt", n_poses=5, overwrite=True)
            w.writerow([cid, name, rid, scores[0], scores[1], scores[2], dist,
                        round(time.time()-t,1)])
        except Exception as e:
            w.writerow([cid, name, rid, "ERR","","","", round(time.time()-t,1)])
            print(f"  error {cid} {rid}: {e}")
        f.flush()
        print(f"  {rid} {cid} {name[:25]:<25} e1={scores[0] if 'scores' in dir() else 'ERR'}")
f.close()
print("\n[OK] re-docking done")

# summary table
df = pd.read_csv(out)
df = df[df["e1"] != "ERR"].copy()
df["e1"] = pd.to_numeric(df["e1"])
piv = df.pivot_table(index=["id","name"], columns="receptor", values="e1").reset_index()
piv.columns.name = None
piv = piv.rename(columns={"1BR5":"e1_1BR5","1BR6":"e1_1BR6"})
piv["consistent"] = (piv["e1_1BR5"] < -6.0) & (piv["e1_1BR6"] < -6.0)
piv = piv.sort_values("e1_1BR5")
print("\n=== CONSISTENT HITS (score < -6.0 in both receptors) ===")
print(piv[piv.consistent][["id","name","e1_1BR5","e1_1BR6"]].to_string(index=False))
print("\n=== FULL TABLE ===")
print(piv.to_string(index=False))
piv.to_csv("results_redock_summary.csv", index=False)
