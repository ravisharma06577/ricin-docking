import numpy as np
from Bio.PDB import PDBParser, Superimposer
from rdkit import Chem
from rdkit.Chem import AllChem
from vina import Vina
from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule, RDKitMolCreate

SM = {"1BR6": "Nc1nc2ncc(CNc3ccc(cc3)C(=O)O)nc2c(=O)[nH]1",   # pteroic acid
      "1BR5": "Nc1nc2ncc(nc2c(=O)[nH]1)C(O)C(O)CO"}            # neopterin
NAME = {"1BR6": "pteroic acid", "1BR5": "neopterin"}

def ca(pid):
    ch = next(PDBParser(QUIET=True).get_structure(pid, f"protein/{pid}_clean.pdb")[0].get_chains())
    return {r.id[1]: r["CA"] for r in ch if "CA" in r}

def ref_in_frame(home, target):
    m = AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles(SM[home]),
                                             Chem.MolFromPDBFile(f"protein/{home}_ligand.pdb"))
    Chem.SanitizeMol(m); m = Chem.RemoveHs(m)
    xyz = m.GetConformer().GetPositions()
    if home != target:
        a, b = ca(target), ca(home)
        common = sorted(set(a) & set(b))
        sup = Superimposer()
        sup.set_atoms([a[i] for i in common], [b[i] for i in common])
        rot, tran = sup.rotran
        xyz = xyz @ rot + tran
        print(f"  (superposition {home}->{target}: {len(common)} CA atoms, RMS {sup.rms:.2f} A)")
    return m, xyz

print("receptor | ligand | seed | pose1 E | pose1 RMSD | best-of-top5 RMSD | best pose rank")
for rec_id in ("1BR6", "1BR5"):
    rec = f"protein/{rec_id}_rec.pdbqt"
    box = np.loadtxt(f"protein/{rec_id}_box.txt")
    for lig_id in ("1BR6", "1BR5"):
        tag = "SELF " if lig_id == rec_id else "CROSS"
        ref, ref_xyz = ref_in_frame(lig_id, rec_id)
        tmpl = Chem.MolFromSmiles(SM[lig_id])
        lig = Chem.AddHs(tmpl)
        AllChem.EmbedMolecule(lig, randomSeed=7); AllChem.MMFFOptimizeMolecule(lig)
        pdbqt, ok, err = PDBQTWriterLegacy.write_string(MoleculePreparation().prepare(lig)[0])
        assert ok, err

        def rmsd(rm, cid):
            xyz = rm.GetConformer(cid).GetPositions()
            return min(float(np.sqrt(((xyz - ref_xyz[list(mt)])**2).sum(1).mean()))
                       for mt in ref.GetSubstructMatches(rm, uniquify=False, maxMatches=5000))

        for seed in (1, 2, 3):
            v = Vina(sf_name="vina", cpu=0, seed=seed, verbosity=0)
            v.set_receptor(rec); v.set_ligand_from_string(pdbqt)
            v.compute_vina_maps(center=box[0].tolist(), box_size=[16.0] * 3)
            v.dock(exhaustiveness=64, n_poses=20)
            E = v.energies(n_poses=20)
            pm = PDBQTMolecule(v.poses(n_poses=20), is_dlg=False, skip_typing=True)
            mols = Chem.RemoveHs(RDKitMolCreate.from_pdbqt_mol(pm)[0])
            r = [rmsd(mols, c) for c in range(mols.GetNumConformers())]
            k = int(np.argmin(r[:5]))
            print(f"{rec_id} | {NAME[lig_id]:<12} {tag} | {seed} | {E[0][0]:.2f} | {r[0]:5.2f} A | {r[k]:.2f} A | #{k+1}")
