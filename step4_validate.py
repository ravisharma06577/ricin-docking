import os, subprocess, requests
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select
from openmm.app import PDBFile
from pdbfixer import PDBFixer
from rdkit import Chem
from rdkit.Chem import AllChem
from vina import Vina
from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule, RDKitMolCreate

KEY = {81, 121, 123, 177, 180}
SYSTEMS = {
    "1BR6": "Nc1nc2ncc(CNc3ccc(cc3)C(=O)O)nc2c(=O)[nH]1",   # pteroic acid
    "1BR5": "Nc1nc2ncc(nc2c(=O)[nH]1)C(O)C(O)CO",            # neopterin
}
STACK_MAX, NRES_MIN = 5.0, 3

def prep(pid):
    P = f"protein/{pid}"
    os.makedirs("protein", exist_ok=True)
    if not os.path.exists(P + ".pdb"):
        r = requests.get(f"https://files.rcsb.org/download/{pid}.pdb", timeout=60)
        r.raise_for_status(); open(P + ".pdb", "w").write(r.text)
    if os.path.exists(P + "_rec.pdbqt") and os.path.exists(P + "_H.pdb") \
            and os.path.exists(P + "_ligand.pdb") and os.path.exists(P + "_box.txt"):
        return P
    model = PDBParser(QUIET=True).get_structure(pid, P + ".pdb")[0]
    hets = [res for ch in model for res in ch if res.id[0].strip() and res.get_resname() != "HOH"]
    lig = max(hets, key=len)
    print(f"[{pid}] ligand detected: {lig.get_resname()} ({len(lig)} atoms)")
    class Prot(Select):
        def accept_residue(self, r): return r.id[0] == " "
    class Lig(Select):
        def accept_residue(self, r): return r is lig
    io = PDBIO(); io.set_structure(model.get_parent())
    io.save(P + "_clean.pdb", Prot()); io.save(P + "_ligand.pdb", Lig())
    c = np.array([a.coord for a in lig]).mean(axis=0)
    np.savetxt(P + "_box.txt", np.vstack([c, [16.0] * 3]), fmt="%.3f")
    fx = PDBFixer(filename=P + "_clean.pdb")
    fx.findMissingResidues(); fx.missingResidues = {}
    fx.findNonstandardResidues(); fx.replaceNonstandardResidues()
    fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.4)
    with open(P + "_H.pdb", "w") as f:
        PDBFile.writeFile(fx.topology, fx.positions, f, keepIds=True)
    subprocess.run(["mk_prepare_receptor.py", "--read_pdb", P + "_H.pdb",
                    "-o", P + "_rec", "--write_pdbqt"], check=True, capture_output=True)
    assert os.path.exists(P + "_rec.pdbqt"), "receptor build failed"
    return P

def receptor_features(P):
    ch = next(PDBParser(QUIET=True).get_structure("r", P + "_H.pdb")[0].get_chains())
    pol, t80 = [], None
    for res in ch:
        n = res.id[1]
        if n in KEY:
            pol += [(n, a.coord) for a in res if a.element in ("N", "O")]
        if n == 80:
            t80 = np.mean([res[x].coord for x in ("CG","CD1","CD2","CE1","CE2","CZ")], axis=0)
    assert t80 is not None, "Tyr80 not found"
    return t80, pol

def feats(mol, cid, t80, pol):
    try: Chem.SanitizeMol(mol)
    except Exception: pass
    xyz = mol.GetConformer(cid).GetPositions()
    rings = [r for r in mol.GetRingInfo().AtomRings()
             if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    stack = min(np.linalg.norm(xyz[list(r)].mean(0) - t80) for r in rings) if rings else 99.0
    het = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in (7, 8)]
    res = {n for n, c in pol if any(np.linalg.norm(xyz[i] - c) <= 3.5 for i in het)}
    return stack, len(res)

for pid, smi in SYSTEMS.items():
    print(f"\n================ {pid} ================")
    P = prep(pid)
    rec = P + "_rec.pdbqt"
    t80, pol = receptor_features(P)
    tmpl = Chem.MolFromSmiles(smi)
    ref = AllChem.AssignBondOrdersFromTemplate(tmpl, Chem.MolFromPDBFile(P + "_ligand.pdb"))
    Chem.SanitizeMol(ref); ref = Chem.RemoveHs(ref)
    ref_xyz = ref.GetConformer().GetPositions()
    s, n = feats(ref, 0, t80, pol)
    print(f"crystal ligand: stack {s:.1f} A, key residues contacted {n}  "
          f"-> rule {'PASS' if s <= STACK_MAX and n >= NRES_MIN else 'FAIL'}")

    lig = Chem.AddHs(tmpl)
    AllChem.EmbedMolecule(lig, randomSeed=7); AllChem.MMFFOptimizeMolecule(lig)
    pdbqt, ok, err = PDBQTWriterLegacy.write_string(MoleculePreparation().prepare(lig)[0])
    assert ok, err
    box = np.loadtxt(P + "_box.txt")

    def rmsd(rm, cid):
        xyz = rm.GetConformer(cid).GetPositions()
        return min(float(np.sqrt(((xyz - ref_xyz[list(m)])**2).sum(1).mean()))
                   for m in ref.GetSubstructMatches(rm, uniquify=False, maxMatches=5000))

    for seed in (1, 2, 3):
        v = Vina(sf_name="vina", cpu=0, seed=seed, verbosity=0)
        v.set_receptor(rec); v.set_ligand_from_string(pdbqt)
        v.compute_vina_maps(center=box[0].tolist(), box_size=[16.0] * 3)
        v.dock(exhaustiveness=64, n_poses=20)
        E = v.energies(n_poses=20)
        pm = PDBQTMolecule(v.poses(n_poses=20), is_dlg=False, skip_typing=True)
        mols = Chem.RemoveHs(RDKitMolCreate.from_pdbqt_mol(pm)[0])
        rows = []
        for c in range(mols.GetNumConformers()):
            st, nr = feats(mols, c, t80, pol)
            rows.append((c + 1, E[c][0], rmsd(mols, c), st, nr, st <= STACK_MAX and nr >= NRES_MIN))
        if seed == 1:
            print("\nseed 1 detail: pose | E | RMSD | stack | residues | rule")
            for r in rows[:10]:
                print(f"  {r[0]:>2} | {r[1]:.2f} | {r[2]:5.2f} | {r[3]:4.1f} | {r[4]} | {'PASS' if r[5] else '-'}")
        best = min(rows, key=lambda r: r[2])
        passing = [r for r in rows if r[5]]
        first = passing[0] if passing else None
        print(f"seed {seed}: vina pose1 RMSD {rows[0][2]:.2f} | closest pose = #{best[0]} ({best[2]:.2f} A) | "
              + (f"first rule-passing pose = #{first[0]} (RMSD {first[2]:.2f} A)" if first else "no pose passes rule"))
