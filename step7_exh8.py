import os, sys, csv, time
import numpy as np
import pandas as pd
from vina import Vina

REC_ID, EXH, NPOSES = "1BR5", 8, 5
rec = f"protein/{REC_ID}_rec.pdbqt"
box = np.loadtxt(f"protein/{REC_ID}_box.txt")
os.makedirs("library/poses_exh8", exist_ok=True)
out_csv = "screen_1BR5_exh8.csv"

lib = pd.read_csv("library/library.csv")
lib = lib[lib["prepared"]].copy()
lib["ctrl"] = lib["id"].str.startswith("CTRL_")
lib = pd.concat([lib[lib.ctrl], lib[~lib.ctrl]]).reset_index(drop=True)
limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(lib)
lib = lib.head(limit)

done = set(pd.read_csv(out_csv)["id"]) if os.path.exists(out_csv) else set()
print(f"{len(lib)} ligands queued, {len(done)} already done")

# crystal ligand centroid (neopterin) to report whether poses sit in the pocket
xyz = [[float(l[30:38]), float(l[38:46]), float(l[46:54])]
       for l in open(f"protein/{REC_ID}_ligand.pdb") if l.startswith("HETATM")]
pocket = np.mean(xyz, axis=0)

def centroid(pdbqt):
    c = [[float(l[30:38]), float(l[38:46]), float(l[46:54])]
         for l in pdbqt.splitlines() if l.startswith(("ATOM", "HETATM"))]
    return np.mean(c, axis=0)

v = Vina(sf_name="vina", cpu=0, seed=42, verbosity=0)
v.set_receptor(rec)
v.compute_vina_maps(center=box[0].tolist(), box_size=[16.0] * 3)

new = not os.path.exists(out_csv)
f = open(out_csv, "a", newline="")
w = csv.writer(f)
if new:
    w.writerow(["id", "name", "heavy", "rotb", "flexible", "e1", "e2", "e3", "e4", "e5",
                "pose1_dist_to_pocket", "seconds"])

t0 = time.time(); n = 0
for _, r in lib.iterrows():
    if r["id"] in done:
        continue
    t = time.time()
    try:
        v.set_ligand_from_file(f"library/pdbqt/{r['id']}.pdbqt")
        v.dock(exhaustiveness=EXH, n_poses=NPOSES)
        E = [round(float(e[0]), 3) for e in v.energies(n_poses=NPOSES)]
        E += [""] * (NPOSES - len(E))
        d = centroid(v.poses(n_poses=1))
        dist = round(float(np.linalg.norm(d - pocket)), 2)
        v.write_poses(f"library/poses_exh8/{r['id']}.pdbqt", n_poses=NPOSES, overwrite=True)
        w.writerow([r["id"], r["name"], r["heavy"], r["rotb"], int(r["rotb"] > 15), *E,
                    dist, round(time.time() - t, 1)])
    except Exception as e:
        w.writerow([r["id"], r["name"], r["heavy"], r["rotb"], int(r["rotb"] > 15),
                    "ERR", "", "", "", "", "", round(time.time() - t, 1)])
        print("  error:", r["id"], str(e)[:80])
    f.flush(); n += 1
    if n % 25 == 0:
        el = time.time() - t0
        print(f"  {n} done this session, {el/n:.1f} s/ligand avg, "
              f"~{(len(lib)-len(done)-n)*el/n/3600:.1f} h left")
f.close()
print("[OK] screen finished or limit reached")
