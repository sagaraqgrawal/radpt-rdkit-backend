"""
RADPT RDKit backend
===================
A tiny Flask API that computes REAL molecular descriptors from a SMILES string
using RDKit. Deployed on Render.com (free tier). RADPT calls POST /descriptors.

Everything this returns is genuinely calculated from structure — not AI-estimated.
The ADMET/tox endpoints RDKit cannot compute stay on the AI side in RADPT and are
labelled separately.

Endpoints
---------
GET  /               -> health check
POST /descriptors    -> { "smiles": "CC(=O)Oc1ccccc1C(=O)O" }
                        returns a JSON block of RDKit-computed descriptors + rule filters
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED, Lipinski
from rdkit.Chem import FilterCatalog

app = Flask(__name__)
# Allow browser calls from your site. Lock this down to your domain in production
# by replacing "*" with "https://radgent.com".
CORS(app, resources={r"/*": {"origins": "*"}})

# Pre-build PAINS + Brenk structural-alert catalogs once at startup.
_pains_params = FilterCatalog.FilterCatalogParams()
_pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
_PAINS = FilterCatalog.FilterCatalog(_pains_params)

_brenk_params = FilterCatalog.FilterCatalogParams()
_brenk_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
_BRENK = FilterCatalog.FilterCatalog(_brenk_params)


def _round(x, n=3):
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return None


def compute(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid": False, "error": "RDKit could not parse this SMILES"}

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    heavy = mol.GetNumHeavyAtoms()
    fcsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    mr = Crippen.MolMR(mol)
    formal_charge = Chem.GetFormalCharge(mol)
    heteroatoms = rdMolDescriptors.CalcNumHeteroatoms(mol)
    n_no_count = Lipinski.NOCount(mol)
    n_nhoh_count = Lipinski.NHOHCount(mol)
    qed_val = QED.qed(mol)

    # ---- Rule-based drug-likeness filters (all computed from the descriptors above) ----
    lipinski_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    veber_pass = (rotb <= 10) and (tpsa <= 140)
    # Egan: TPSA <= 131.6 and -1 <= WLOGP(≈MolLogP) <= 5.88
    egan_pass = (tpsa <= 131.6) and (-1.0 <= logp <= 5.88)
    # Ghose: 160<=MW<=480, -0.4<=logP<=5.6, 40<=MR<=130, 20<=atoms<=70
    ghose_pass = (160 <= mw <= 480) and (-0.4 <= logp <= 5.6) and (40 <= mr <= 130) and (20 <= heavy <= 70)
    # Muegge: MW 200-600, TPSA<=150, logP -2..5, rings<=7, rotb<=15, HBA<=10, HBD<=5, arom_rings<=... etc (simplified core)
    muegge_pass = (200 <= mw <= 600) and (-2 <= logp <= 5) and (tpsa <= 150) and (rings <= 7) \
        and (rotb <= 15) and (hba <= 10) and (hbd <= 5)

    # ---- Structural alerts ----
    pains_matches = [e.GetDescription() for e in _PAINS.GetMatches(mol)]
    brenk_matches = [e.GetDescription() for e in _BRENK.GetMatches(mol)]

    # ---- Lead-likeness (common heuristic) ----
    lead_like = (250 <= mw <= 350) and (logp <= 3.5) and (rotb <= 7)

    return {
        "valid": True,
        "engine": "RDKit " + Chem.rdBase.rdkitVersion,
        "canonical_smiles": Chem.MolToSmiles(mol),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "physchem": {
            "mw": _round(mw, 2),
            "exact_mw": _round(Descriptors.ExactMolWt(mol), 4),
            "logP": _round(logp),
            "tpsa": _round(tpsa, 2),
            "hbd": hbd,
            "hba": hba,
            "rotatable_bonds": rotb,
            "aromatic_rings": arom_rings,
            "ring_count": rings,
            "heavy_atoms": heavy,
            "heteroatoms": heteroatoms,
            "fraction_csp3": _round(fcsp3),
            "molar_refractivity": _round(mr, 2),
            "formal_charge": formal_charge,
            "no_count": n_no_count,
            "nhoh_count": n_nhoh_count,
        },
        "druglikeness": {
            "lipinski": {"pass": lipinski_violations == 0, "violations": lipinski_violations},
            "veber": {"pass": bool(veber_pass)},
            "egan": {"pass": bool(egan_pass)},
            "ghose": {"pass": bool(ghose_pass)},
            "muegge": {"pass": bool(muegge_pass)},
            "lead_likeness": {"pass": bool(lead_like)},
        },
        "medchem": {
            "qed": _round(qed_val, 3),
        },
        "alerts": {
            "pains": {"count": len(pains_matches), "matches": pains_matches[:8]},
            "brenk": {"count": len(brenk_matches), "matches": brenk_matches[:8]},
        },
    }


@app.route("/")
def health():
    return jsonify({"status": "ok", "service": "RADPT RDKit backend",
                    "rdkit": Chem.rdBase.rdkitVersion})


@app.route("/descriptors", methods=["POST"])
def descriptors():
    data = request.get_json(silent=True) or {}
    smiles = (data.get("smiles") or "").strip()
    if not smiles:
        return jsonify({"valid": False, "error": "no smiles provided"}), 400
    try:
        return jsonify(compute(smiles))
    except Exception as e:  # never 500 the browser; return a clean error
        return jsonify({"valid": False, "error": str(e)}), 200


if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
