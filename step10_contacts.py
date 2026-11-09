import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from meeko import PDBQTMolecule, RDKitMolCreate
RDLogger.DisableLog("rdApp.*")

HITS = [
    ("CHEMBL416",  "Methoxsalen"),
    ("CHEMBL290916","Edaravone"),
    ("CHEMBL887",  "Rasagiline"),
    ("CHEMBL590",  "Menadione"),
    ("CHEMBL1177", "Pemoline"),
    ("CHEMBL770",  "Tolazoline"),
    ("CHEMBL818",  "Iobenguane"),
    ("CHEMBL1467", "Allopurinol"),
]
KEY = {78:"ASN78",80:"TYR80",81:"VAL81",93:"PHE93",
       121:"GLY121",122:"ASN122",123:"TYR123",172:"ILE172",
       177:"GLU177",180:"ARG180",208:"GLU208",211:"TRP211"}
CUT = 4.0

def res_atoms(pdb_path):
    ch = next(PDBParser(QUIET=True).get_structure("r",pdb_path)[0].get_chains())
    return {n: np.array([a.coord for a in res])
            for res in ch if (n:=res.id[1]) in KEY}

RA = res_atoms("protein/1BR5_H.pdb")
RB = res_atoms("protein/1BR6_H.pdb")

print(f"{'Drug':<22} {'Receptor':<8} {'Contacts (≤4Å)'}")
print("-"*75)
for cid, name in HITS:
    for recid, RA_ in [("1BR5", RA), ("1BR6", RB)]:
        pdbqt = f"redock/{cid}_{recid}.pdbqt"
        try:
            pm  = PDBQTMolecule.from_file(pdbqt, skip_typing=True)
            mol = RDKitMolCreate.from_pdbqt_mol(pm)[0]
            if mol is None: raise ValueError("mol is None")
            mol = Chem.RemoveHs(mol)
            xyz = mol.GetConformer(0).GetPositions()
        except Exception as e:
            print(f"  {name:<22} {recid:<8} (pose read error: {e})")
            continue
        contacts = []
        for rnum, ratoms in RA_.items():
            d = np.min(np.linalg.norm(ratoms[:,None,:] - xyz[None,:,:], axis=2))
            if d <= CUT:
                contacts.append(f"{KEY[rnum]}({d:.1f})")
        print(f"  {name:<22} {recid:<8} {', '.join(contacts) if contacts else 'none'}")
    print()
