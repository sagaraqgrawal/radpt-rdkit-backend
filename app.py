"""
RADPT RDKit backend — v2
========================
Flask API computing REAL, deterministic molecular descriptors from SMILES via RDKit.
Nothing here is AI-estimated. Every value is reproducible from the molecular graph.

CHANGES FROM v1
---------------
1. FIXED  Muegge filter — the `or True` clause made the aromatic-ring criterion a
          no-op, and the carbon/heteroatom count criteria were missing entirely.
2. FIXED  Abbott bioavailability score — v1 could never emit 0.17 or 0.85.
          (STILL FLAGGED: verify branch thresholds against Martin 2005. See note.)
3. ADDED  Synthetic accessibility (SA) score — RDKit contrib sascorer
          (Ertl & Schuffenhauer 2009). Previously this came from the AI layer.
4. FIXED  Dimorphite-DL now fails LOUDLY. v1 silently swallowed an import error and
          dropped the ionisation block. A `_status` object now reports what is live.
5. ADDED  Descriptors recomputed on the pH-7.4 microspecies, so logP-vs-logD and
          neutral-vs-zwitterion effects are visible rather than hidden.
6. ADDED  /batch endpoint — analyse a whole analogue series in one call.
7. ADDED  Morgan fingerprint (bit string) in the response, so applicability-domain
          / Tanimoto analysis can be done offline without re-deriving structures.

Endpoints
---------
GET  /            -> health check + component status
POST /descriptors -> { "smiles": "CC(=O)Oc1ccccc1C(=O)O" }
POST /batch       -> { "compounds": [ {"name": "...", "smiles": "..."}, ... ] }
"""

import os
import sys

from flask import Flask, request, jsonify
from flask_cors import CORS

from rdkit import Chem
from rdkit.Chem import (
    Descriptors, Crippen, rdMolDescriptors, QED, Lipinski, GraphDescriptors,
    RDConfig
)
from rdkit.Chem import FilterCatalog
from rdkit.Chem import rdFingerprintGenerator  # noqa: F401  (kept for other uses)

# Morgan/ECFP4 generator lives in ad.py so the applicability-domain reference
# fingerprints and the query fingerprint are guaranteed to be constructed identically.
import ad
_MORGAN_GEN = ad.MORGAN_GEN

# ----------------------------------------------------------------------------
# Synthetic accessibility score (Ertl & Schuffenhauer 2009).
# Ships with RDKit under Contrib/SA_Score but is not importable by default.
# ----------------------------------------------------------------------------
_SA_ERR = None
try:
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer  # noqa: E402
    _HAS_SASCORER = True
except Exception as e:          # pragma: no cover
    _HAS_SASCORER = False
    _SA_ERR = str(e)

# ----------------------------------------------------------------------------
# Dimorphite-DL — protonation state at a given pH.
# The API changed between major versions, so try each known entry point and
# RECORD which one worked. v1 of this backend swallowed failures silently, which
# meant the ionisation block could vanish without anyone noticing. For zwitterions
# (betalains, amino acids, catecholamines) that field is the whole point, so a
# silent failure is worse than a crash.
# ----------------------------------------------------------------------------
_DIMORPHITE_MODE = None
_DIMORPHITE_ERR = None
_protonate_fn = None

try:
    # dimorphite-dl >= 2.x
    from dimorphite_dl import protonate_smiles as _dd_protonate

    def _protonate_fn(smi, ph):                     # noqa: F811
        return list(_dd_protonate(smi, ph_min=ph, ph_max=ph))

    _DIMORPHITE_MODE = "protonate_smiles (v2 API)"
except Exception as e_v2:
    _DIMORPHITE_ERR = f"v2: {e_v2}"
    try:
        # dimorphite-dl 1.3.x
        from dimorphite_dl import DimorphiteDL as _DD

        _dd = _DD(min_ph=7.4, max_ph=7.4, pka_precision=1.0)

        def _protonate_fn(smi, ph):                 # noqa: F811
            _dd.min_ph = ph
            _dd.max_ph = ph
            return list(_dd.protonate(smi))

        _DIMORPHITE_MODE = "DimorphiteDL class (v1.3 API)"
    except Exception as e_v1:
        _DIMORPHITE_ERR += f" | v1.3: {e_v1}"

_HAS_DIMORPHITE = _protonate_fn is not None

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ----------------------------------------------------------------------------
# Structural-alert catalogues (built once at startup)
# ----------------------------------------------------------------------------
def _make_catalog(*cats):
    p = FilterCatalog.FilterCatalogParams()
    for c in cats:
        p.AddCatalog(c)
    return FilterCatalog.FilterCatalog(p)


_C = FilterCatalog.FilterCatalogParams.FilterCatalogs
_PAINS      = _make_catalog(_C.PAINS)
_BRENK      = _make_catalog(_C.BRENK)
_NIH        = _make_catalog(_C.NIH)
_SURECHEMBL = _make_catalog(_C.CHEMBL_SureChEMBL)
_ZINC       = _make_catalog(_C.ZINC)
_CHEMBL     = _make_catalog(_C.CHEMBL)


def _round(x, n=3):
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return None


# ============================================================================
# RULE SETS
# ============================================================================

def _muegge(mw, logp, tpsa, rings, rotb, hba, hbd, n_carbon, n_hetero):
    """
    Muegge (Bayer) drug-likeness filter.
    Muegge, Heald & Brittelli (2001) J Med Chem 44:1841-6.

    v1 BUG: the final clause was `(n_arom_heavy // 6 <= 4 or True)` — `or True`
    short-circuits, so the aromatic-ring criterion never applied. The carbon-count
    and heteroatom-count criteria were absent altogether. All seven criteria below
    are now enforced.
    """
    return (
        200 <= mw <= 600
        and -2 <= logp <= 5
        and tpsa <= 150
        and rings <= 7
        and n_carbon > 4
        and n_hetero > 1
        and rotb <= 15
        and hba <= 10
        and hbd <= 5
    )


def _abbott_bioavailability(charge, tpsa, mw, logp, hbd, hba):
    """
    Abbott Bioavailability Score — Martin (2005) J Med Chem 48:3164-70.
    Estimates P(F > 10% in rat).

    v1 BUG: the neutral/cationic failure branch returned 0.11, so the score could
    never emit 0.17, and the anionic branch was collapsed to a single threshold.

    >>> VERIFY THIS AGAINST MARTIN (2005) BEFORE PUBLISHING ANY VALUE FROM IT. <<<
    The branch structure below reflects the scheme as commonly implemented
    (e.g. SwissADME), whose permissible outputs are 0.11, 0.17, 0.55, 0.56, 0.85.
    Sagar: read the primary paper, confirm the TPSA cut-offs, and delete this
    warning once you have. Do not publish a number you have not traced to source.
    """
    if charge < 0:                       # anionic
        if tpsa < 75:
            return 0.85
        if tpsa < 150:
            return 0.56
        return 0.11
    # neutral, cationic or zwitterionic -> Lipinski-based branch
    ro5_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return 0.55 if ro5_violations == 0 else 0.17


# ============================================================================
# IONISATION
# ============================================================================

def _ionization_at_ph(smiles, neutral_charge, ph=7.4):
    """Dominant protonation state at a given pH via Dimorphite-DL (Ropp et al. 2019)."""
    if not _HAS_DIMORPHITE:
        return {"available": False, "reason": _DIMORPHITE_ERR}
    try:
        forms = _protonate_fn(smiles, ph)
        if not forms:
            return {"available": False, "reason": "no protonation states returned"}
        prot = forms[0]
        m = Chem.MolFromSmiles(prot)
        if m is None:
            return {"available": False, "reason": "protonated SMILES did not parse"}

        charge = Chem.GetFormalCharge(m)

        # Zwitterion detection: net charge may be 0 while formal +/- centres coexist.
        pos = sum(1 for a in m.GetAtoms() if a.GetFormalCharge() > 0)
        neg = sum(1 for a in m.GetAtoms() if a.GetFormalCharge() < 0)
        if pos and neg:
            cls = "Zwitterionic" if charge == 0 else (
                "Zwitterionic, net anionic" if charge < 0 else "Zwitterionic, net cationic"
            )
        elif charge < 0:
            cls = "Anionic"
        elif charge > 0:
            cls = "Cationic"
        else:
            cls = "Neutral"

        return {
            "available": True,
            "protonated_smiles": prot,
            f"net_charge_pH{ph}": charge,
            "positive_centres": pos,
            "negative_centres": neg,
            "class": cls,
            "all_states": forms[:6],
            "engine": _DIMORPHITE_MODE,
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}


def _microspecies_descriptors(prot_smiles):
    """
    Recompute the descriptors most affected by ionisation on the pH-7.4 microspecies.

    Why this matters: cLogP is computed on the NEUTRAL structure and is therefore a
    partition coefficient, not logD. For a zwitterion the two diverge enormously,
    and quoting cLogP as though it described membrane partitioning at physiological
    pH is a real and common error. Reporting both makes the gap visible.
    """
    m = Chem.MolFromSmiles(prot_smiles)
    if m is None:
        return None
    return {
        "formal_charge": Chem.GetFormalCharge(m),
        "tpsa": _round(rdMolDescriptors.CalcTPSA(m), 2),
        "hbd": rdMolDescriptors.CalcNumHBD(m),
        "hba": rdMolDescriptors.CalcNumHBA(m),
        "clogp_of_microspecies": _round(Crippen.MolLogP(m)),
        "_caveat": (
            "Crippen cLogP is parameterised on neutral, drug-like molecules. Applied "
            "to a charged microspecies it is an extrapolation and is NOT a validated "
            "logD. Report it as such or not at all."
        ),
    }


# ============================================================================
# MAIN COMPUTATION
# ============================================================================

def compute(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid": False, "error": "RDKit could not parse this SMILES"}

    # --- core descriptors ---
    mw         = Descriptors.MolWt(mol)
    exact_mw   = Descriptors.ExactMolWt(mol)
    heavy_mw   = Descriptors.HeavyAtomMolWt(mol)
    logp       = Crippen.MolLogP(mol)
    mr         = Crippen.MolMR(mol)
    tpsa       = rdMolDescriptors.CalcTPSA(mol)
    labute_asa = rdMolDescriptors.CalcLabuteASA(mol)

    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    hba  = rdMolDescriptors.CalcNumHBA(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)

    arom_rings  = rdMolDescriptors.CalcNumAromaticRings(mol)
    aliph_rings = rdMolDescriptors.CalcNumAliphaticRings(mol)
    sat_rings   = rdMolDescriptors.CalcNumSaturatedRings(mol)
    rings       = rdMolDescriptors.CalcNumRings(mol)
    heavy       = mol.GetNumHeavyAtoms()
    heteroatoms = rdMolDescriptors.CalcNumHeteroatoms(mol)
    fcsp3       = rdMolDescriptors.CalcFractionCSP3(mol)
    formal_charge = Chem.GetFormalCharge(mol)
    n_stereo        = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    n_unspec_stereo = rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
    spiro       = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    bridgehead  = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    n_amide     = rdMolDescriptors.CalcNumAmideBonds(mol)
    n_no        = Lipinski.NOCount(mol)
    n_nhoh      = Lipinski.NHOHCount(mol)
    n_arom_heavy = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    n_carbon     = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")

    ring_sizes = [len(r) for r in mol.GetRingInfo().AtomRings()]
    max_ring = max(ring_sizes) if ring_sizes else 0

    bertz   = _round(GraphDescriptors.BertzCT(mol), 1)
    qed_val = QED.qed(mol)

    # --- synthetic accessibility (NEW — was AI-estimated in v1) ---
    if _HAS_SASCORER:
        sa = {
            "value": _round(sascorer.calculateScore(mol), 2),
            "scale": "1 (easily made) to 10 (very hard)",
            "source": "RDKit Contrib sascorer — Ertl & Schuffenhauer (2009)",
        }
    else:
        sa = {"value": None, "error": _SA_ERR}

    # --- rule-based drug-likeness filters ---
    lip_v = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    veber = (rotb <= 10) and (tpsa <= 140)
    egan  = (tpsa <= 131.6) and (-1.0 <= logp <= 5.88)
    ghose = (160 <= mw <= 480) and (-0.4 <= logp <= 5.6) \
        and (40 <= mr <= 130) and (20 <= heavy <= 70)
    muegge = _muegge(mw, logp, tpsa, rings, rotb, hba, hbd, n_carbon, heteroatoms)

    pfizer_flag = (logp > 3) and (tpsa < 75)      # True = FLAGGED (tox risk)
    gsk         = (mw <= 400) and (logp <= 4)
    golden_tri  = (200 <= mw <= 450) and (-2 <= logp <= 5)
    lead_like   = (250 <= mw <= 350) and (logp <= 3.5) and (rotb <= 7)

    abbott = _abbott_bioavailability(formal_charge, tpsa, mw, logp, hbd, hba)

    # --- structural alerts ---
    def _matches(cat):
        m = [e.GetDescription() for e in cat.GetMatches(mol)]
        return {"count": len(m), "matches": m[:8]}

    # --- ionisation at pH 7.4 + microspecies descriptors ---
    ionization = _ionization_at_ph(smiles, formal_charge, 7.4)
    microspecies = None
    if ionization.get("available") and ionization.get("protonated_smiles"):
        microspecies = _microspecies_descriptors(ionization["protonated_smiles"])

    # --- Morgan fingerprint (for offline applicability-domain / Tanimoto work) ---
    fp = _MORGAN_GEN.GetFingerprint(mol)
    fp_bits = list(fp.GetOnBits())

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
            "_logP_note": "Wildman-Crippen cLogP on the NEUTRAL structure. This is logP, not logD.",
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
            "carbon_atoms": n_carbon,
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
            "lipinski":          {"pass": lip_v == 0, "violations": lip_v},
            "veber":             {"pass": bool(veber)},
            "egan":              {"pass": bool(egan)},
            "ghose":             {"pass": bool(ghose)},
            "muegge":            {"pass": bool(muegge)},
            "gsk_4_400":         {"pass": bool(gsk)},
            "golden_triangle":   {"pass": bool(golden_tri)},
            "pfizer_3_75_flag":  {"flagged": bool(pfizer_flag)},
            "lead_likeness":     {"pass": bool(lead_like)},
            "bioavailability_score": abbott,
        },
        "medchem": {
            "qed": _round(qed_val, 3),
            "sa_score": sa,
        },
        "alerts": {
            "pains":        _matches(_PAINS),
            "brenk":        _matches(_BRENK),
            "nih":          _matches(_NIH),
            "surechembl":   _matches(_SURECHEMBL),
            "zinc":         _matches(_ZINC),
            "chembl_firms": _matches(_CHEMBL),
        },
        "ionization_pH74": ionization,
        "microspecies_pH74": microspecies,
        "applicability_domain": ad.assess(mol),
        "fingerprint": {
            "type": "Morgan/ECFP4, radius 2, 2048 bits",
            "on_bits": fp_bits,
        },
    }


# ============================================================================
# ROUTES
# ============================================================================

def _status():
    return {
        "rdkit": Chem.rdBase.rdkitVersion,
        "sascorer": {"available": _HAS_SASCORER, "error": _SA_ERR},
        "dimorphite_dl": {
            "available": _HAS_DIMORPHITE,
            "mode": _DIMORPHITE_MODE,
            "error": None if _HAS_DIMORPHITE else _DIMORPHITE_ERR,
        },
        "applicability_domain": ad.status(),
    }


@app.route("/")
def health():
    return jsonify({
        "status": "ok",
        "service": "RADPT RDKit backend v2",
        "components": _status(),
    })


@app.route("/descriptors", methods=["POST"])
def descriptors():
    data = request.get_json(silent=True) or {}
    smiles = (data.get("smiles") or "").strip()
    if not smiles:
        return jsonify({"valid": False, "error": "no smiles provided"}), 400
    try:
        out = compute(smiles)
        out["_status"] = _status()
        return jsonify(out)
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)}), 200


@app.route("/batch", methods=["POST"])
def batch():
    """
    Analyse a whole analogue series in one call.
    POST { "compounds": [ {"name": "Betanin", "smiles": "..."}, ... ] }
    Returns results in the same order, each tagged with its name.
    """
    data = request.get_json(silent=True) or {}
    compounds = data.get("compounds") or []
    if not isinstance(compounds, list) or not compounds:
        return jsonify({"error": "provide a non-empty 'compounds' list"}), 400
    if len(compounds) > 200:
        return jsonify({"error": "max 200 compounds per batch"}), 400

    results = []
    for c in compounds:
        name = (c.get("name") or "").strip() or "unnamed"
        smi = (c.get("smiles") or "").strip()
        if not smi:
            results.append({"name": name, "valid": False, "error": "no smiles"})
            continue
        try:
            r = compute(smi)
        except Exception as e:
            r = {"valid": False, "error": str(e)}
        r["name"] = name
        r["input_smiles"] = smi
        results.append(r)

    return jsonify({"count": len(results), "results": results, "_status": _status()})


if __name__ == "__main__":
    print("RADPT backend v2 — component status:", _status(), flush=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
