import os, time, json, requests
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from meeko import MoleculePreparation, PDBQTWriterLegacy
RDLogger.DisableLog("rdApp.*")

os.makedirs("library/pdbqt", exist_ok=True)

# ---- 1. download approved small molecules from ChEMBL ----
cache = "library/chembl_approved_raw.json"
if os.path.exists(cache):
    recs = json.load(open(cache))
else:
    url = ("https://www.ebi.ac.uk/chembl/api/data/molecule.json"
           "?max_phase=4&molecule_type=Small%20molecule&limit=1000")
    recs, base = [], "https://www.ebi.ac.uk"
    while url:
        for attempt in range(4):
            try:
                r = requests.get(url, timeout=120); r.raise_for_status(); break
            except Exception as e:
                print("  retry", attempt + 1, e); time.sleep(5)
        else:
            raise SystemExit("ChEMBL download failed - paste this output to me.")
        js = r.json()
        recs += js["molecules"]
        nxt = js["page_meta"]["next"]
        url = base + nxt if nxt else None
        print(f"  downloaded {len(recs)} records")
    json.dump(recs, open(cache, "w"))
print(f"[OK] {len(recs)} ChEMBL records")

# ---- 2. standardize ----
chooser = rdMolStandardize.LargestFragmentChooser()
unch = rdMolStandardize.Uncharger()
rows, seen = [], set()
for m in recs:
    st = m.get("molecule_structures")
    if not st or not st.get("canonical_smiles"):
        continue
    mol = Chem.MolFromSmiles(st["canonical_smiles"])
    if mol is None:
        continue
    mol = unch.uncharge(chooser.choose(mol))
    ha = mol.GetNumHeavyAtoms()
    if not (10 <= ha <= 70):
        continue
    key = Chem.MolToInchiKey(mol)[:14]
    if key in seen:
        continue
    seen.add(key)
    rows.append(dict(id=m["molecule_chembl_id"], name=m.get("pref_name") or "",
                     smiles=Chem.MolToSmiles(mol), heavy=ha,
                     rotb=rdMolDescriptors.CalcNumRotatableBonds(mol),
                     mw=round(Descriptors.MolWt(mol), 1)))
# controls (same SMILES as validation)
rows += [dict(id="CTRL_pteroic_acid", name="pteroic acid (control)",
              smiles="Nc1nc2ncc(CNc3ccc(cc3)C(=O)O)nc2c(=O)[nH]1", heavy=23, rotb=4, mw=313.3),
         dict(id="CTRL_neopterin", name="neopterin (control)",
              smiles="Nc1nc2ncc(nc2c(=O)[nH]1)C(O)C(O)CO", heavy=18, rotb=3, mw=253.2)]
df = pd.DataFrame(rows).drop_duplicates("id").reset_index(drop=True)
print(f"[OK] {len(df)} unique molecules after cleanup (10-70 heavy atoms)")

# ---- 3. 3D + PDBQT ----
prep = MoleculePreparation()
ok_ids, fails = [], []
for i, r in df.iterrows():
    out = f"library/pdbqt/{r['id']}.pdbqt"
    if os.path.exists(out):
        ok_ids.append(r["id"]); continue
    try:
        mol = Chem.AddHs(Chem.MolFromSmiles(r["smiles"]))
        if AllChem.EmbedMolecule(mol, randomSeed=7, maxAttempts=50) != 0:
            if AllChem.EmbedMolecule(mol, randomSeed=7, useRandomCoords=True) != 0:
                raise ValueError("embed failed")
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
        except Exception:
            AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
        s, ok, err = PDBQTWriterLegacy.write_string(prep.prepare(mol)[0])
        if not ok: raise ValueError(err)
        open(out, "w").write(s)
        ok_ids.append(r["id"])
    except Exception as e:
        fails.append((r["id"], str(e)[:60]))
    if (i + 1) % 250 == 0:
        print(f"  prepared {i+1}/{len(df)}  (failures so far: {len(fails)})")

df["prepared"] = df["id"].isin(ok_ids)
df.to_csv("library/library.csv", index=False)
print(f"\n[OK] {len(ok_ids)} ligands ready, {len(fails)} failed")
print("failed examples:", fails[:5])
print(df[["heavy", "rotb", "mw"]].describe().round(1))
