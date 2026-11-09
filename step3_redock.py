import os, subprocess
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from openmm.app import PDBFile
from pdbfixer import PDBFixer
from vina import Vina
from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule, RDKitMolCreate

P = "protein/1BR6"

# ---- 1. add hydrogens to receptor (pH 7.4) ----
fixer = PDBFixer(filename=P + "_clean.pdb")
fixer.findMissingResidues(); fixer.missingResidues = {}
fixer.findNonstandardResidues(); fixer.replaceNonstandardResidues()
fixer.findMissingAtoms(); fixer.addMissingAtoms()
fixer.addMissingHydrogens(7.4)
with open(P + "_H.pdb", "w") as f:
    PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
print("[OK] hydrogens added")

# ---- 2. receptor PDBQT via Meeko (command-line flags differ between versions) ----
rec = P + "_rec.pdbqt"
variants = [
    ["mk_prepare_receptor.py", "-i", P + "_H.pdb", "-o", P + "_rec", "-p"],
    ["mk_prepare_receptor.py", "--read_pdb", P + "_H.pdb", "-o", P + "_rec", "--write_pdbqt"],
    ["mk_prepare_receptor", "--read_pdb", P + "_H.pdb", "-o", P + "_rec", "--write_pdbqt"],
]
for cmd in variants:
    if os.path.exists(rec): break
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception as e:
        print("  tried:", " ".join(cmd[:2]), "-> failed")
if not os.path.exists(rec):
    raise SystemExit("Receptor PDBQT not created - paste the output above to me.")
print("[OK] receptor:", rec)

# ---- 3. reference (crystal) ligand with correct bond orders ----
smiles = "Nc1nc2ncc(CNc3ccc(cc3)C(=O)O)nc2c(=O)[nH]1"   # pteroic acid
tmpl = Chem.MolFromSmiles(smiles)
ref = Chem.MolFromPDBFile(P + "_ligand.pdb")
ref = AllChem.AssignBondOrdersFromTemplate(tmpl, ref)
ref = Chem.RemoveHs(ref)
ref_xyz = ref.GetConformer().GetPositions()

# ---- 4. fresh 3D ligand (NOT the crystal pose) ----
lig = Chem.AddHs(tmpl)
AllChem.EmbedMolecule(lig, randomSeed=7)
AllChem.MMFFOptimizeMolecule(lig)
setups = MoleculePreparation().prepare(lig)
pdbqt, ok, err = PDBQTWriterLegacy.write_string(setups[0])
assert ok, err

def rmsd(rm, cid):
    xyz = rm.GetConformer(cid).GetPositions()
    best = 1e9
    for m in ref.GetSubstructMatches(rm, uniquify=False, maxMatches=5000):
        d = xyz - ref_xyz[list(m)]
        best = min(best, float(np.sqrt((d ** 2).sum(1).mean())))
    return best

# ---- 5. dock with 3 seeds ----
box = np.loadtxt(P + "_box.txt")
print("\nseed | best score | RMSD pose1 | best RMSD in top5")
for seed in (1, 2, 3):
    v = Vina(sf_name="vina", cpu=0, seed=seed)
    v.set_receptor(rec)
    v.set_ligand_from_string(pdbqt)
    v.compute_vina_maps(center=box[0].tolist(), box_size=box[1].tolist())
    v.dock(exhaustiveness=32, n_poses=10)
    E = v.energies(n_poses=10)
    pm = PDBQTMolecule(v.poses(n_poses=10), is_dlg=False, skip_typing=True)
    rm = Chem.RemoveHs(RDKitMolCreate.from_pdbqt_mol(pm)[0])
    r = [rmsd(rm, c) for c in range(rm.GetNumConformers())]
    print(f"  {seed}  |  {E[0][0]:.2f} kcal/mol  |  {r[0]:.2f} A  |  {min(r[:5]):.2f} A")
print("\nPass criterion: pose-1 RMSD <= 2.0 A in most seeds.")
