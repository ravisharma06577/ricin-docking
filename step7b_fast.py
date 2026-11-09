import os, sys, csv, time
import numpy as np
import pandas as pd
from multiprocessing import Pool
from vina import Vina

REC_ID, EXH, NPOSES, NWORK = "1BR5", 8, 5, 8
rec = f"protein/{REC_ID}_rec.pdbqt"
box = np.loadtxt(f"protein/{REC_ID}_box.txt")
out_csv = "screen_fast_1BR5.csv"
os.makedirs("library/poses_fast", exist_ok=True)

xyz = [[float(l[30:38]), float(l[38:46]), float(l[46:54])]
       for l in open(f"protein/{REC_ID}_ligand.pdb") if l.startswith("HETATM")]
pocket = np.mean(xyz, axis=0)
_v = None

def init():
    global _v
    _v = Vina(sf_name="vina", cpu=1, seed=42, verbosity=0)
    _v.set_receptor(rec)
    _v.compute_vina_maps(center=box[0].tolist(), box_size=[16.0] * 3)

def work(a):
    cid, name, heavy, rotb = a
    t = time.time()
    try:
        _v.set_ligand_from_file(f"library/pdbqt/{cid}.pdbqt")
        _v.dock(exhaustiveness=EXH, n_poses=NPOSES)
        E = [round(float(e[0]), 3) for e in _v.energies(n_poses=NPOSES)]
        E += [""] * (NPOSES - len(E))
        p = _v.poses(n_poses=1)
        c = np.mean([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                     for l in p.splitlines() if l.startswith(("ATOM", "HETATM"))], axis=0)
        _v.write_poses(f"library/poses_fast/{cid}.pdbqt", n_poses=NPOSES, overwrite=True)
        return [cid, name, heavy, rotb, *E, round(float(np.linalg.norm(c - pocket)), 2),
                round(time.time() - t, 1)]
    except Exception as e:
        return [cid, name, heavy, rotb, "ERR", "", "", "", "", "", round(time.time() - t, 1)]

if __name__ == "__main__":
    lib = pd.read_csv("library/library.csv")
    lib = lib[lib["prepared"]].copy()
    lib["ctrl"] = lib["id"].str.startswith("CTRL_")
    flex = lib[(lib.rotb > 15) & ~lib.ctrl]
    flex.to_csv("library/deferred_flexible.csv", index=False)
    lib = lib.drop(flex.index)
    lib = pd.concat([lib[lib.ctrl], lib[~lib.ctrl]]).reset_index(drop=True)
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(lib)
    lib = lib.head(limit)

    done = set(pd.read_csv(out_csv)["id"]) if os.path.exists(out_csv) else set()
    todo = [(r.id, r["name"], r.heavy, r.rotb) for _, r in lib.iterrows() if r["id"] not in done]
    print(f"{len(flex)} very flexible drugs deferred; {len(todo)} to dock now, {len(done)} already done")

    new = not os.path.exists(out_csv)
    f = open(out_csv, "a", newline=""); w = csv.writer(f)
    if new:
        w.writerow(["id", "name", "heavy", "rotb", "e1", "e2", "e3", "e4", "e5", "pose1_dist_to_pocket", "seconds"])
    t0 = time.time()
    with Pool(NWORK, initializer=init) as pool:
        for i, row in enumerate(pool.imap_unordered(work, todo, chunksize=1), 1):
            w.writerow(row); f.flush()
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)} done | {el/i:.1f} s/ligand wall | ~{(len(todo)-i)*el/i/60:.0f} min left")
    f.close()
    print("[OK] finished")
