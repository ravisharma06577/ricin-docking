import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from vina import Vina
from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule, RDKitMolCreate

P = "protein/1BR6"
rec = P + "_rec.pdbqt"

smiles = "Nc1nc2ncc(CNc3ccc(cc3)C(=O)O)nc2c(=O)[nH]1"   # pteroic acid
tmpl = Chem.MolFromSmiles(smiles)
ref = AllChem.AssignBondOrdersFromTemplate(tmpl, Chem.MolFromPDBFile(P + "_ligand.pdb"))
ref = Chem.RemoveHs(ref)
ref_xyz = ref.GetConformer().GetPositions()

lig = Chem.AddHs(tmpl)
AllChem.EmbedMolecule(lig, randomSeed=7)
AllChem.MMFFOptimizeMolecule(lig)
pdbqt, ok, err = PDBQTWriterLegacy.write_string(MoleculePreparation().prepare(lig)[0])
assert ok, err

def rmsd(rm, cid):
    xyz = rm.GetConformer(cid).GetPositions()
    best = 1e9
    for m in ref.GetSubstructMatches(rm, uniquify=False, maxMatches=5000):
        d = xyz - ref_xyz[list(m)]
        best = min(best, float(np.sqrt((d ** 2).sum(1).mean())))
    return best

box = np.loadtxt(P + "_box.txt")
center = box[0].tolist()

print("box | seed | pose1 E | pose1 RMSD | best-RMSD pose: rank, RMSD, E")
for size in (20.0, 16.0):
    for seed in (1, 2, 3, 4, 5):
        v = Vina(sf_name="vina", cpu=0, seed=seed, verbosity=0)
        v.set_receptor(rec)
        v.set_ligand_from_string(pdbqt)
        v.compute_vina_maps(center=center, box_size=[size] * 3)
        v.dock(exhaustiveness=64, n_poses=20)
        E = v.energies(n_poses=20)
        pm = PDBQTMolecule(v.poses(n_poses=20), is_dlg=False, skip_typing=True)
        rm = Chem.RemoveHs(RDKitMolCreate.from_pdbqt_mol(pm)[0])
        r = [rmsd(rm, c) for c in range(rm.GetNumConformers())]
        k = int(np.argmin(r))
        print(f"{size:>4.0f} |  {seed}   | {E[0][0]:.2f} | {r[0]:5.2f} A | rank {k+1:>2}, {r[k]:.2f} A, {E[k][0]:.2f}")
