import os, requests
import numpy as np
from collections import Counter
from Bio.PDB import PDBParser, PDBIO, Select

PDB_ID = "1BR6"
os.makedirs("protein", exist_ok=True)
raw = f"protein/{PDB_ID}.pdb"
if not os.path.exists(raw):
    r = requests.get(f"https://files.rcsb.org/download/{PDB_ID}.pdb", timeout=60)
    r.raise_for_status()
    open(raw, "w").write(r.text)

model = PDBParser(QUIET=True).get_structure(PDB_ID, raw)[0]

print("== Non-water HETATM groups ==")
hets = []
for ch in model:
    for res in ch:
        if res.id[0].strip() and res.get_resname() != "HOH":
            hets.append((res.get_resname(), ch.id, res.id[1], len(res), res))
            print(f"{res.get_resname()} chain {ch.id} resnum {res.id[1]} atoms={len(res)}")

if not hets:
    raise SystemExit("No ligand found - stopping.")

# largest HETATM group = the inhibitor (ions/sulfate are small)
name, cid, rnum, natoms, lig = max(hets, key=lambda h: h[3])
print(f"\nUsing ligand: {name} (chain {cid}, res {rnum}, {natoms} heavy atoms)")

coords = np.array([a.coord for a in lig])
center = coords.mean(axis=0)
extent = coords.max(axis=0) - coords.min(axis=0)
size = extent + 12.0          # 6 A padding each side
size = np.maximum(size, 20.0) # minimum 20 A box
print(f"Box center (x,y,z): {center.round(3).tolist()}")
print(f"Ligand extent:      {extent.round(2).tolist()}")
print(f"Box size:           {size.round(1).tolist()}")

class Prot(Select):
    def accept_residue(self, r): return r.id[0] == " "
class Lig(Select):
    def accept_residue(self, r): return r is lig

io = PDBIO(); io.set_structure(model.get_parent())
io.save(f"protein/{PDB_ID}_clean.pdb", Prot())
io.save(f"protein/{PDB_ID}_ligand.pdb", Lig())

np.savetxt(f"protein/{PDB_ID}_box.txt",
           np.vstack([center, size]), fmt="%.3f",
           header="row1=center xyz, row2=size xyz")

print("\nActive-site contacts within 4.5 A of ligand:")
near = Counter()
for ch in model:
    for res in ch:
        if res.id[0] == " ":
            d = min(np.linalg.norm(a.coord - b.coord) for a in res for b in lig)
            if d <= 4.5:
                near[(res.id[1], res.get_resname())] = round(d, 1)
for (n, rn), d in sorted(near.items()):
    print(f"  {rn}{n}  {d} A")
print(f"\n[OK] saved protein/{PDB_ID}_clean.pdb, _ligand.pdb, _box.txt")
