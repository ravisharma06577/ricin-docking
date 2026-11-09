import os, numpy as np
from Bio.PDB import PDBParser

CANDIDATES = {
    "CHEMBL416":  "Methoxsalen",
    "CHEMBL290916": "Edaravone",
    "CHEMBL887":  "Rasagiline",
    "CHEMBL1177": "Pemoline",
    "CHEMBL590":  "Menadione",
    "CHEMBL1467": "Allopurinol",
}
KEY = {78,80,81,93,121,122,123,172,177,180,208,211}

for rid in ("1BR5","1BR6"):
    ch = next(PDBParser(QUIET=True).get_structure(
              rid,f"protein/{rid}_H.pdb")[0].get_chains())
    res_xyz = {r.id[1]: {a.get_name(): a.coord for a in r}
               for r in ch if r.id[1] in KEY}
    print(f"\n{'='*60}")
    print(f"Receptor {rid}")
    for cid, name in CANDIDATES.items():
        pose = f"redock/{cid}_{rid}.pdbqt"
        if not os.path.exists(pose):
            print(f"  {name}: pose file missing"); continue
        lig_xyz = np.array([[float(l[30:38]),float(l[38:46]),float(l[46:54])]
                   for l in open(pose) if l.startswith(("ATOM","HETATM"))
                   and "BRANCH" not in l and "ROOT" not in l])
        contacts = []
        for rnum, atoms in res_xyz.items():
            d = min(np.linalg.norm(np.array(c)-lx)
                    for c in atoms.values() for lx in lig_xyz)
            if d <= 4.0:
                contacts.append((rnum, round(d,1)))
        contacts.sort()
        print(f"  {name:<15} contacts: {', '.join(f'{r}({d}A)' for r,d in contacts)}")

# PyMOL session commands
print("\n\n=== PyMOL commands (run in PyMOL terminal) ===")
print("# paste these into PyMOL after: File > Open protein/1BR5_clean.pdb")
for cid, name in CANDIDATES.items():
    slug = name.lower().replace(" ","_")
    print(f"load redock/{cid}_1BR5.pdbqt, {slug}")
print("hide everything")
print("show cartoon, 1BR5_clean")
print("show sticks, resn PTR+HOH and chain A")
for cid, name in CANDIDATES.items():
    slug = name.lower().replace(" ","_")
    print(f"show sticks, {slug}")
print("color white, 1BR5_clean")
print("color cyan, methoxsalen")
print("color yellow, edaravone")
print("color orange, rasagiline")
print("color magenta, pemoline")
print("color green, menadione")
print("color red, allopurinol")
print("select pocket, resi 78+80+81+93+121+122+123+172+177+180+208+211")
print("show sticks, pocket")
print("color tv_green, pocket")
print("zoom pocket")
print("ray 1200,900")
print("png figures/figure1_pocket_overview.png, dpi=300")
os.makedirs("figures", exist_ok=True)
print("\nfigures/ directory created for output.")
