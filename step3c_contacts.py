import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem
from vina import Vina
from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule, RDKitMolCreate

P = "protein/1BR6"
rec = P + "_rec.pdbqt"
KEY = {81, 121, 123, 177, 180}          # H-bond partners in the adenine pocket
CORE = Chem.MolFromSmarts("[#7]~[#6]1~[#7]~[#6]2~[#7]~[#6]~[#6]~[#7]~[#6]2~[#6](~[#8])~[#7]1")

# receptor atoms
rm_ = next(PDBParser(QUIET=True).get_structure("r", P + "_H.pdb")[0].get_chains())
key_xyz, tyr80 = [], None
for res in rm_:
    n = res.id[1]
    if n in KEY:
        key_xyz += [a.coord for a in res if a.element in ("N", "O")]
    if n == 80:
        tyr80 = np.mean([res[x].coord for x in ("CG","CD1","CD2","CE1","CE2","CZ")], axis=0)
key_xyz = np.array(key_xyz)

def features(mol, cid=0):
    xyz = mol.GetConformer(cid).GetPositions()
    match = mol.GetSubstructMatch(CORE)
    if not match:
        return None
    centroid = xyz[list(match)].mean(axis=0)
    stack = np.linalg.norm(centroid - tyr80)
    het = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in (7, 8)]
    contacts = sum(1 for i in het if np.linalg.norm(key_xyz - xyz[i], axis=1).min() <= 3.5)
    return stack, contacts

tmpl = Chem.MolFromSmiles("Nc1nc2ncc(CNc3ccc(cc3)C(=O)O)nc2c(=O)[nH]1")
ref = AllChem.AssignBondOrdersFromTemplate(tmpl, Chem.MolFromPDBFile(P + "_ligand.pdb"))
ref = Chem.RemoveHs(ref)
ref_xyz = ref.GetConformer().GetPositions()
print("CRYSTAL reference  (pterin-Tyr80 dist, polar contacts):", features(ref))

lig = Chem.AddHs(tmpl)
AllChem.EmbedMolecule(lig, randomSeed=7); AllChem.MMFFOptimizeMolecule(lig)
pdbqt, ok, err = PDBQTWriterLegacy.write_string(MoleculePreparation().prepare(lig)[0])
assert ok, err

def rmsd(rm, cid):
    xyz = rm.GetConformer(cid).GetPositions()
    return min(float(np.sqrt(((xyz - ref_xyz[list(m)])**2).sum(1).mean()))
               for m in ref.GetSubstructMatches(rm, uniquify=False, maxMatches=5000))

box = np.loadtxt(P + "_box.txt")
v = Vina(sf_name="vina", cpu=0, seed=1, verbosity=0)
v.set_receptor(rec); v.set_ligand_from_string(pdbqt)
v.compute_vina_maps(center=box[0].tolist(), box_size=[16.0]*3)
v.dock(exhaustiveness=64, n_poses=20)
E = v.energies(n_poses=20)
pm = PDBQTMolecule(v.poses(n_poses=20), is_dlg=False, skip_typing=True)
mols = Chem.RemoveHs(RDKitMolCreate.from_pdbqt_mol(pm)[0])

print("\npose | energy | RMSD to crystal | pterin-Tyr80 (A) | polar contacts")
for c in range(mols.GetNumConformers()):
    f = features(mols, c)
    s = f"{f[0]:.1f} | {f[1]}" if f else "core not matched"
    print(f"{c+1:>4} | {E[c][0]:.2f} | {rmsd(mols, c):5.2f} A | {s}")
