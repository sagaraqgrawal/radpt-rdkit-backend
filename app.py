"""
RADPT RDKit backend
===================
Flask API that computes REAL molecular descriptors from a SMILES string using RDKit.
Everything returned here is genuinely calculated from structure (deterministic),
not AI-estimated. RADPT labels these values with a green "RDKIT" chip.

Endpoints
---------
GET  /               -> health check
POST /descriptors    -> { "smiles": "CC(=O)Oc1ccccc1C(=O)O" }
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

from rdkit import Chem
from rdkit.Chem import (
    Descriptors, Crippen, rdMolDescriptors, QED, Lipinski, GraphDescriptors
)
from rdkit.Chem import FilterCatalog

# Optional: dimorphite-dl for real protonation-state (ionization) prediction at pH 7.4.
# If it isn't installed, the backend still runs and just omits the ionization block.
try:
    from dimorphite_dl import protonate_smiles as _protonate
    _HAS_DIMORPHITE = True
except Exception:
    _HAS_DIMORPHITE = False

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Structural-alert catalogs (built once at startup)
def _make_catalog(*cats):
    p = FilterCatalog.FilterCatalogParams()
    for c in cats:
        p.AddCatalog(c)
    return FilterCatalog.FilterCatalog(p)

_C = FilterCatalog.FilterCatalogParams.FilterCatalogs
_PAINS = _make_catalog(_C.PAINS)
_BRENK = _make_catalog(_C.BRENK)
_NIH   = _make_catalog(_C.NIH)
_SURECHEMBL = _make_catalog(_C.CHEMBL_SureChEMBL)
_ZINC  = _make_catalog(_C.ZINC)
_CHEMBL = _make_catalog(_C.CHEMBL)   # aggregate of the medchem-firm alert sets


def _round(x, n=3):
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return None


def _abbott_bioavailability(mol, tpsa, logp, mw, hbd, hba, rotb):
    """Abbott bioavailability score (0.55 if passes a simplified rule set, else 0.11)."""
    # Simplified Martin 2005 heuristic used by SwissADME-style scoring.
    charge = Chem.GetFormalCharge(mol)
    if charge < 0:
        return 0.11 if not (tpsa <= 150 and mw <= 500) else 0.56
    # neutral/positive path — pass if Ro5-ish and TPSA<=150
    passes = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10 and rotb <= 10 and tpsa <= 150)
    return 0.55 if passes else 0.11


def _ionization_at_ph74(smiles, neutral_charge):
    """Real protonation-state prediction at pH 7.4 via dimorphite-dl (if available)."""
    if not _HAS_DIMORPHITE:
        return None
    try:
        forms = _protonate(smiles, ph_min=7.4, ph_max=7.4)
        if not forms:
            return None
        prot = forms[0]
        m = Chem.MolFromSmiles(prot)
        if m is None:
            return None
        charge = Chem.GetFormalCharge(m)
        if charge < neutral_charge:
            cls = "Anionic (acidic)"
        elif charge > neutral_charge:
            cls = "Cationic (basic)"
        elif charge < 0:
            cls = "Anionic"
        elif charge > 0:
            cls = "Cationic"
        else:
            cls = "Neutral"
        return {"protonated_smiles": prot, "net_charge_pH7.4": charge, "class": cls}
    except Exception:
        return None


def compute(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid": False, "error": "RDKit could not parse this SMILES"}

    # --- core descriptors ---
    mw       = Descriptors.MolWt(mol)
    exact_mw = Descriptors.ExactMolWt(mol)
    heavy_mw = Descriptors.HeavyAtomMolWt(mol)
    logp     = Crippen.MolLogP(mol)
    mr       = Crippen.MolMR(mol)
    tpsa     = rdMolDescriptors.CalcTPSA(mol)
    labute_asa = rdMolDescriptors.CalcLabuteASA(mol)

    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    hba  = rdMolDescriptors.CalcNumHBA(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)

    arom_rings   = rdMolDescriptors.CalcNumAromaticRings(mol)
    aliph_rings  = rdMolDescriptors.CalcNumAliphaticRings(mol)
    sat_rings    = rdMolDescriptors.CalcNumSaturatedRings(mol)
    rings        = rdMolDescriptors.CalcNumRings(mol)
    heavy        = mol.GetNumHeavyAtoms()
    heteroatoms  = rdMolDescriptors.CalcNumHeteroatoms(mol)
    fcsp3        = rdMolDescriptors.CalcFractionCSP3(mol)
    formal_charge= Chem.GetFormalCharge(mol)
    n_stereo     = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    n_unspec_stereo = rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
    spiro        = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    bridgehead   = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    n_amide      = rdMolDescriptors.CalcNumAmideBonds(mol)
    n_no    = Lipinski.NOCount(mol)
    n_nhoh  = Lipinski.NHOHCount(mol)
    n_arom_heavy = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())

    # largest ring size
    ring_sizes = [len(r) for r in mol.GetRingInfo().AtomRings()]
    max_ring = max(ring_sizes) if ring_sizes else 0

    # topological / complexity
    bertz = _round(GraphDescriptors.BertzCT(mol), 1)
    qed_val = QED.qed(mol)

    # --- rule-based drug-likeness filters ---
    lip_v = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    veber = (rotb <= 10) and (tpsa <= 140)
    egan  = (tpsa <= 131.6) and (-1.0 <= logp <= 5.88)
    ghose = (160 <= mw <= 480) and (-0.4 <= logp <= 5.6) and (40 <= mr <= 130) and (20 <= heavy <= 70)
    muegge = (200 <= mw <= 600) and (-2 <= logp <= 5) and (tpsa <= 150) and (rings <= 7) \
        and (rotb <= 15) and (hba <= 10) and (hbd <= 5) and (n_arom_heavy // 6 <= 4 or True)
    # Pfizer 3/75 flag: risk if logp>3 AND tpsa<75 (True = flagged/risky)
    pfizer_flag = (logp > 3) and (tpsa < 75)
    # GSK 4/400: pass if mw<=400 and logp<=4
    gsk = (mw <= 400) and (logp <= 4)
    # Golden Triangle: 200<=MW<=450 and -2<=logD(≈logP)<=5
    golden_tri = (200 <= mw <= 450) and (-2 <= logp <= 5)
    lead_like = (250 <= mw <= 350) and (logp <= 3.5) and (rotb <= 7)

    abbott = _abbott_bioavailability(mol, tpsa, logp, mw, hbd, hba, rotb)

    # --- structural alerts ---
    pains = [e.GetDescription() for e in _PAINS.GetMatches(mol)]
    brenk = [e.GetDescription() for e in _BRENK.GetMatches(mol)]
    nih   = [e.GetDescription() for e in _NIH.GetMatches(mol)]
    surechembl = [e.GetDescription() for e in _SURECHEMBL.GetMatches(mol)]
    zinc  = [e.GetDescription() for e in _ZINC.GetMatches(mol)]
    chembl = [e.GetDescription() for e in _CHEMBL.GetMatches(mol)]

    # --- ionization state at pH 7.4 (real, dimorphite) ---
    ionization = _ionization_at_ph74(smiles, formal_charge)

    return {
        "valid": True,
        "engine": "RDKit " + Chem.rdBase.rdkitVersion,
        "canonical_smiles": Chem.MolToSmiles(mol),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "physchem": {
            "mw": _round(mw, 2),
            "exact_mw": _round(exact_mw, 4),
            "heavy_atom_mw": _round(heavy_mw, 2),
            "logP": _round(logp),
            "molar_refractivity": _round(mr, 2),
            "tpsa": _round(tpsa, 2),
            "labute_asa": _round(labute_asa, 2),
            "hbd": hbd,
            "hba": hba,
            "rotatable_bonds": rotb,
            "aromatic_rings": arom_rings,
            "aliphatic_rings": aliph_rings,
            "saturated_rings": sat_rings,
            "ring_count": rings,
            "max_ring_size": max_ring,
            "heavy_atoms": heavy,
            "aromatic_heavy_atoms": n_arom_heavy,
            "heteroatoms": heteroatoms,
            "fraction_csp3": _round(fcsp3),
            "formal_charge": formal_charge,
            "stereocenters": n_stereo,
            "unspecified_stereocenters": n_unspec_stereo,
            "spiro_atoms": spiro,
            "bridgehead_atoms": bridgehead,
            "amide_bonds": n_amide,
            "no_count": n_no,
            "nhoh_count": n_nhoh,
            "bertz_complexity": bertz,
        },
        "druglikeness": {
            "lipinski":     {"pass": lip_v == 0, "violations": lip_v},
            "veber":        {"pass": bool(veber)},
            "egan":         {"pass": bool(egan)},
            "ghose":        {"pass": bool(ghose)},
            "muegge":       {"pass": bool(muegge)},
            "gsk_4_400":    {"pass": bool(gsk)},
            "golden_triangle": {"pass": bool(golden_tri)},
            "pfizer_3_75_flag": {"flagged": bool(pfizer_flag)},
            "lead_likeness": {"pass": bool(lead_like)},
            "bioavailability_score": abbott,
        },
        "medchem": {
            "qed": _round(qed_val, 3),
        },
        "alerts": {
            "pains": {"count": len(pains), "matches": pains[:8]},
            "brenk": {"count": len(brenk), "matches": brenk[:8]},
            "nih":   {"count": len(nih),   "matches": nih[:8]},
            "surechembl": {"count": len(surechembl), "matches": surechembl[:8]},
            "zinc":  {"count": len(zinc),  "matches": zinc[:8]},
            "chembl_firms": {"count": len(chembl), "matches": chembl[:8]},
        },
        "ionization_pH74": ionization,
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
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)}), 200


if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
