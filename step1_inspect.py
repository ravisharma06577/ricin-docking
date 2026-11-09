import os, requests
from collections import Counter
from Bio.PDB import PDBParser, PDBIO, Select

PDB_ID = "1RTC"
os.makedirs("protein", exist_ok=True)
raw = f"protein/{PDB_ID}.pdb"

if not os.path.exists(raw):
    r = requests.get(f"https://files.rcsb.org/download/{PDB_ID}.pdb", timeout=60)
    r.raise_for_status()
    open(raw, "w").write(r.text)
print(f"[OK] {raw} ready")

model = PDBParser(QUIET=True).get_structure(PDB_ID, raw)[0]

print("\n== Chains (standard residues) ==")
for ch in model:
    n = sum(1 for r in ch if r.id[0] == " ")
    print(f"chain {ch.id}: {n} residues")

print("\n== Non-protein entries (HETATM) ==")
het = Counter()
for ch in model:
    for r in ch:
        if r.id[0].strip() and r.get_resname() != "HOH":
            het[(r.get_resname(), ch.id, r.id[1])] += 1
waters = sum(1 for r in model.get_residues() if r.get_resname() == "HOH")
for k in sorted(het): print(f"{k[0]} chain {k[1]} resnum {k[2]}")
print(f"waters: {waters}")

print("\n== Active-site residue check (expected names/numbers) ==")
expected = {80: "TYR", 123: "TYR", 177: "GLU", 180: "ARG", 211: "TRP"}
chain = next(ch for ch in model if any(r.id[0] == " " for r in ch))
for num, name in expected.items():
    try:
        r = chain[(" ", num, " ")]
        ok = "OK" if r.get_resname() == name else "MISMATCH"
        print(f"{name}{num}: found {r.get_resname()} -> {ok}")
    except KeyError:
        print(f"{name}{num}: NOT FOUND")

class ProteinOnly(Select):
    def accept_chain(self, c): return c.id == chain.id
    def accept_residue(self, r): return r.id[0] == " "

io = PDBIO(); io.set_structure(model.get_parent())
io.save(f"protein/{PDB_ID}_clean.pdb", ProteinOnly())
print(f"\n[OK] saved protein/{PDB_ID}_clean.pdb (chain {chain.id}, no water, no ligands)")
