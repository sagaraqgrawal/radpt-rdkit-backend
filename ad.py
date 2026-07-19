"""
RADPT — applicability domain (AD) assessment
============================================
Max/kNN Tanimoto similarity of a query structure to a reference set of approved
small-molecule drugs, using Morgan/ECFP4 fingerprints.

WHY
---
Every predictive layer in RADPT (rule filters, QSAR models, and the AI layer) was
parameterised on drug-like chemical space. When a query sits far outside that space,
the predictions are extrapolations and should be labelled as such. This module makes
that distance explicit and reportable.

METHOD
------
- Fingerprint: Morgan radius 2, 2048 bits (ECFP4). Rogers & Hahn (2010) JCIM 50:742-54.
- Similarity: Tanimoto coefficient to every reference structure.
- Reported: max similarity (nearest neighbour), mean of top-5 (kNN, more robust to a
  single fortuitous match), and the nearest neighbour's name.
- Thresholds: NOT hardcoded. At startup the module computes the leave-one-out
  nearest-neighbour similarity for every reference compound, giving the distribution of
  "how isolated is a typical member of drug space". The 25th and 5th percentiles of that
  distribution become the in-domain and borderline boundaries respectively.

  Rationale: a query is in-domain if it is no more isolated from known drugs than a
  typical known drug is. This self-calibrates to whatever reference set you load, which
  a fixed 0.4 cutoff does not. See Sheridan et al. (2004) JCIM 44:1912-28 and
  Sahigara et al. (2012) Molecules 17:4791-810 for AD methodology generally; the
  percentile calibration here is a distance-to-model variant, not a literature method
  you can cite by name. Describe it accurately in any manuscript.

LIMITATIONS — read before publishing
------------------------------------
1. AD is MODEL-SPECIFIC. This module measures distance to approved-drug space, which is
   a proxy for the domain of RADPT's rule-based filters and any QSAR model trained on
   drug-like data. It is NOT the applicability domain of the LLM layer, which has no
   defined training distribution. Do not present it as validating AI-generated rows.
2. Fingerprint similarity is a structural measure, not a property measure. Two compounds
   with Tanimoto 0.8 can differ by orders of magnitude in solubility (activity cliffs).
   In-domain means "the model has seen things like this", not "the prediction is right".
3. ECFP4 similarity is systematically lower for small molecules (fewer set bits). A
   query with <12 heavy atoms will look extrapolative almost regardless of what it is;
   the module flags this rather than pretending otherwise.
4. The seed reference set is small (~60) and hand-entered. Use build_reference_set.py to
   replace it with a fetched ChEMBL approved-drug set before any publication.
"""

from __future__ import annotations

import os
from statistics import mean

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

# Shared with app.py — import this generator there so query and reference
# fingerprints are guaranteed identical in construction.
MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

FP_SPEC = "Morgan/ECFP4, radius 2, 2048 bits"

# Below this heavy-atom count, ECFP4 Tanimoto is unreliable as a domain measure.
_SMALL_MOL_HEAVY_ATOM_FLOOR = 12

# ----------------------------------------------------------------------------
# ABSOLUTE THRESHOLD FLOORS — do not remove.
#
# The percentile calibration below only behaves sensibly when the reference set is
# DENSE. On a small, structurally diverse set (e.g. the ~77-compound seed set), every
# member's nearest neighbour is far away, so the 25th/5th percentiles collapse to
# ~0.20 — which is merely the background ECFP4 similarity that any two organic
# molecules share. Calibrated alone, the seed set labelled TNT and ethanol as
# "in domain", which is worse than no flag at all.
#
# These floors cap that failure. The effective threshold is always
# max(calibrated, floor), so a dense reference set can raise the bar but a sparse
# one can never lower it below a defensible minimum.
#
# 0.35 is a conventional ECFP4 "structurally related" line; 0.20 is near the noise
# floor. Both are conventions, not derived constants.
# ----------------------------------------------------------------------------
FLOOR_IN_DOMAIN = 0.35
FLOOR_EXTRAPOLATION = 0.20

REFERENCE_FILE = os.environ.get(
    "RADPT_REFERENCE_SET",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_drugs.smi"),
)

# ----------------------------------------------------------------------------
# Seed reference set — approved oral small-molecule drugs spanning common scaffolds.
#
# HAND-ENTERED. Structures were chosen for confidence, but this set has NOT been
# verified against an authoritative database. It is a functional default so the
# module works out of the box, not a publication-grade reference.
#
# Known gaps: no macrolides, no glycosides, no peptidic drugs, no macrocycles.
# That biases the set toward classical Ro5 space, which will make beyond-Ro5 queries
# (including betanin and other betalains) read as extrapolation. That is arguably the
# correct answer, but it is a property of THIS reference set, not a universal truth.
# Replace via build_reference_set.py.
# ----------------------------------------------------------------------------
_SEED_REFERENCE: list[tuple[str, str]] = [
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("paracetamol", "CC(=O)Nc1ccc(O)cc1"),
    ("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("naproxen", "COc1ccc2cc(ccc2c1)C(C)C(=O)O"),
    ("diclofenac", "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl"),
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("theophylline", "Cn1c(=O)c2[nH]cnc2n(C)c1=O"),
    ("metronidazole", "Cc1ncc([N+](=O)[O-])n1CCO"),
    ("fluconazole", "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F"),
    ("ciprofloxacin", "OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O"),
    ("levofloxacin", "C[C@H]1COc2c3n1cc(C(=O)O)c(=O)c3cc(F)c2N1CCN(C)CC1"),
    ("linezolid", "CC(=O)NC[C@H]1CN(c2ccc(N3CCOCC3)c(F)c2)C(=O)O1"),
    ("amoxicillin", "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O"),
    ("ampicillin", "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccccc3)C(=O)N2[C@H]1C(=O)O"),
    ("cefalexin", "CC1=C(N2[C@H](SC1)[C@H](NC(=O)[C@H](N)c1ccccc1)C2=O)C(=O)O"),
    ("trimethoprim", "COc1cc(Cc2cnc(N)nc2N)cc(OC)c1OC"),
    ("sulfamethoxazole", "Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1"),
    ("dapsone", "Nc1ccc(cc1)S(=O)(=O)c1ccc(N)cc1"),
    ("isoniazid", "NNC(=O)c1ccncc1"),
    ("pyrazinamide", "NC(=O)c1cnccn1"),
    ("ethambutol", "CC[C@@H](CO)NCCN[C@@H](CC)CO"),
    ("chloroquine", "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12"),
    ("acyclovir", "Nc1nc2c(ncn2COCCO)c(=O)[nH]1"),
    ("atenolol", "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"),
    ("propranolol", "CC(C)NCC(O)COc1cccc2ccccc12"),
    ("metoprolol", "COCCc1ccc(OCC(O)CNC(C)C)cc1"),
    ("amlodipine", "CCOC(=O)C1=C(COCCN)NC(C)=C(C1c1ccccc1Cl)C(=O)OC"),
    ("nifedipine", "COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1[N+](=O)[O-]"),
    ("verapamil", "COc1ccc(CCN(C)CCCC(C#N)(C(C)C)c2ccc(OC)c(OC)c2)cc1OC"),
    ("diltiazem", "COc1ccc(cc1)[C@@H]1Sc2ccccc2N(CCN(C)C)C(=O)[C@@H]1OC(C)=O"),
    ("losartan", "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1"),
    ("captopril", "CC(CS)C(=O)N1CCC[C@H]1C(=O)O"),
    ("enalapril", "CCOC(=O)[C@H](CCc1ccccc1)N[C@@H](C)C(=O)N1CCC[C@H]1C(=O)O"),
    ("lisinopril", "NCCCC[C@H](N[C@@H](CCc1ccccc1)C(=O)O)C(=O)N1CCC[C@H]1C(=O)O"),
    ("prazosin", "COc1cc2nc(nc(N)c2cc1OC)N1CCN(CC1)C(=O)c1ccco1"),
    ("furosemide", "NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl"),
    ("hydrochlorothiazide", "NS(=O)(=O)c1cc2c(cc1Cl)NCNS2(=O)=O"),
    ("atorvastatin", "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O"),
    ("simvastatin", "CCC(C)(C)C(=O)O[C@H]1C[C@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@@H]12"),
    ("metformin", "CN(C)C(=N)NC(N)=N"),
    ("glibenclamide", "COc1ccc(Cl)cc1C(=O)NCCc1ccc(cc1)S(=O)(=O)NC(=O)NC1CCCCC1"),
    ("omeprazole", "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1"),
    ("pantoprazole", "COc1ccnc(CS(=O)c2nc3ccc(OC(F)F)cc3[nH]2)c1OC"),
    ("ranitidine", "CNC(=C[N+](=O)[O-])NCCSCc1ccc(CN(C)C)o1"),
    ("cimetidine", "CC1=C(CSCCNC(=NC#N)NC)N=CN1"),
    ("ondansetron", "Cc1nccn1CC1CCc2c(C1=O)c1ccccc1n2C"),
    ("loratadine", "CCOC(=O)N1CCC(=C2c3ccc(Cl)cc3CCc3cccnc32)CC1"),
    ("cetirizine", "OC(=O)COCCN1CCN(CC1)C(c1ccccc1)c1ccc(Cl)cc1"),
    ("diphenhydramine", "CN(C)CCOC(c1ccccc1)c1ccccc1"),
    ("salbutamol", "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1"),
    ("diazepam", "CN1c2ccc(Cl)cc2C(c2ccccc2)=NCC1=O"),
    ("fluoxetine", "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1"),
    ("sertraline", "CN[C@H]1CC[C@@H](c2ccc(Cl)c(Cl)c2)c2ccccc21"),
    ("haloperidol", "OC1(CCN(CCCC(=O)c2ccc(F)cc2)CC1)c1ccc(Cl)cc1"),
    ("olanzapine", "Cc1cc2c(s1)Nc1ccccc1N=C2N1CCN(C)CC1"),
    ("risperidone", "Cc1nc2CCCCn2c(=O)c1CCN1CCC(CC1)c1noc2cc(F)ccc12"),
    ("quetiapine", "OCCOCCN1CCN(CC1)C1=Nc2ccccc2Sc2ccccc21"),
    ("carbamazepine", "NC(=O)N1c2ccccc2C=Cc2ccccc21"),
    ("phenytoin", "O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1"),
    ("lamotrigine", "Nc1nnc(-c2cccc(Cl)c2Cl)c(N)n1"),
    ("valproic acid", "CCCC(CCC)C(=O)O"),
    ("gabapentin", "NCC1(CC(=O)O)CCCCC1"),
    ("morphine", "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"),
    ("naloxone", "C=CCN1CC[C@]23c4c5ccc(O)c4O[C@H]2C(=O)CC[C@]3(O)[C@H]1C5"),
    ("tramadol", "COc1cccc(c1)C1(O)CCCCC1CN(C)C"),
    ("warfarin", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"),
    ("clopidogrel", "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1"),
    ("allopurinol", "O=c1[nH]cnc2[nH]ncc12"),
    ("methotrexate", "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(cc1)C(=O)N[C@@H](CCC(=O)O)C(=O)O"),
    ("fluorouracil", "O=c1[nH]cc(F)c(=O)[nH]1"),
    ("tamoxifen", "CC/C(=C(\\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1"),
    ("imatinib", "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"),
    ("gefitinib", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"),
    ("sildenafil", "CCCc1nn(C)c2c1nc([nH]c2=O)-c1cc(ccc1OCC)S(=O)(=O)N1CCN(C)CC1"),
    ("testosterone", "C[C@]12CC[C@H]3[C@@H](CC[C@H]4CC(=O)CC[C@]34C)[C@@H]1CC[C@@H]2O"),
    ("levodopa", "N[C@@H](Cc1ccc(O)c(O)c1)C(=O)O"),
    ("nicotine", "CN1CCC[C@H]1c1cccnc1"),
]


# ----------------------------------------------------------------------------
# Reference set loading
# ----------------------------------------------------------------------------
def _load_reference_records() -> tuple[list[tuple[str, str]], str]:
    """Return (records, provenance_string). Prefers an on-disk .smi file."""
    if os.path.exists(REFERENCE_FILE):
        records = []
        with open(REFERENCE_FILE, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t") if "\t" in line else line.split(None, 1)
                if len(parts) == 1:
                    records.append(("unnamed", parts[0]))
                else:
                    # tolerate either "smiles<TAB>name" or "name<TAB>smiles"
                    a, b = parts[0].strip(), parts[1].strip()
                    if Chem.MolFromSmiles(a) is not None:
                        records.append((b, a))
                    else:
                        records.append((a, b))
        if records:
            return records, f"file: {os.path.basename(REFERENCE_FILE)} ({len(records)} entries)"
    return _SEED_REFERENCE, (
        f"built-in seed set ({len(_SEED_REFERENCE)} approved drugs, hand-entered, "
        "NOT database-verified — replace before publication"
    )


class _ReferenceSet:
    """Fingerprinted reference set with self-calibrated AD thresholds."""

    def __init__(self):
        records, provenance = _load_reference_records()
        self.provenance = provenance
        self.names: list[str] = []
        self.fps = []
        self.rejected: list[str] = []

        for name, smi in records:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                self.rejected.append(name)
                continue
            self.names.append(name)
            self.fps.append(MORGAN_GEN.GetFingerprint(mol))

        self.size = len(self.fps)
        self.loo_nn: list[float] = []
        self.t_in_domain = None
        self.t_borderline = None
        self.calibrated_in_domain = None
        self.calibrated_borderline = None
        self.floor_is_binding = False
        self.error = None

        if self.size < 20:
            self.error = (
                f"reference set has only {self.size} valid structures; "
                "AD thresholds cannot be calibrated reliably (need >=20)"
            )
            return

        self._calibrate()

    def _calibrate(self):
        """Leave-one-out nearest-neighbour similarity distribution."""
        for i, fp in enumerate(self.fps):
            sims = DataStructs.BulkTanimotoSimilarity(fp, self.fps)
            sims[i] = -1.0  # exclude self
            self.loo_nn.append(max(sims))
        self.loo_nn.sort()

        # raw, uncapped percentile values (reported for transparency)
        self.calibrated_in_domain = self._percentile(25)
        self.calibrated_borderline = self._percentile(5)

        # effective thresholds — floors win when the reference set is too sparse
        self.t_in_domain = max(self.calibrated_in_domain, FLOOR_IN_DOMAIN)
        self.t_borderline = max(self.calibrated_borderline, FLOOR_EXTRAPOLATION)

        self.floor_is_binding = (
            self.calibrated_in_domain < FLOOR_IN_DOMAIN
            or self.calibrated_borderline < FLOOR_EXTRAPOLATION
        )

    def _percentile(self, p: float) -> float:
        """Linear-interpolated percentile of the sorted LOO NN distribution."""
        if not self.loo_nn:
            return 0.0
        k = (len(self.loo_nn) - 1) * (p / 100.0)
        lo, hi = int(k), min(int(k) + 1, len(self.loo_nn) - 1)
        if lo == hi:
            return round(self.loo_nn[lo], 4)
        frac = k - lo
        return round(self.loo_nn[lo] * (1 - frac) + self.loo_nn[hi] * frac, 4)

    def percentile_rank(self, value: float) -> float:
        """What fraction of reference compounds are MORE isolated than this query."""
        if not self.loo_nn:
            return 0.0
        below = sum(1 for v in self.loo_nn if v < value)
        return round(100.0 * below / len(self.loo_nn), 1)


_REF = _ReferenceSet()


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def status() -> dict:
    """Component status, for the /health endpoint."""
    return {
        "available": _REF.error is None,
        "error": _REF.error,
        "reference_set_size": _REF.size,
        "reference_provenance": _REF.provenance,
        "rejected_entries": _REF.rejected,
        "fingerprint": FP_SPEC,
        "thresholds": {
            "in_domain_at_or_above": _REF.t_in_domain,
            "extrapolation_below": _REF.t_borderline,
            "calibrated_in_domain": _REF.calibrated_in_domain,
            "calibrated_extrapolation": _REF.calibrated_borderline,
            "absolute_floors": {
                "in_domain": FLOOR_IN_DOMAIN,
                "extrapolation": FLOOR_EXTRAPOLATION,
            },
            "floor_is_binding": _REF.floor_is_binding,
            "calibration": (
                "max(25th/5th percentile of leave-one-out NN similarity within the "
                "reference set, absolute floor)"
            ),
        },
        "warning": (
            "Reference set is too sparse for percentile calibration to be meaningful; "
            "absolute floors are governing the verdicts. Replace with a larger set "
            "(build_reference_set.py) for calibration to do real work."
        ) if _REF.floor_is_binding else None,
    }


def assess(mol, k: int = 5) -> dict:
    """
    Assess a query molecule against the reference set.

    Returns a dict safe to embed in the /descriptors response. Never raises —
    on any failure it returns {"available": False, "reason": ...} so AD never
    blocks descriptor computation.
    """
    if _REF.error is not None:
        return {"available": False, "reason": _REF.error}
    if mol is None:
        return {"available": False, "reason": "no molecule"}

    try:
        fp = MORGAN_GEN.GetFingerprint(mol)
        sims = DataStructs.BulkTanimotoSimilarity(fp, _REF.fps)
        if not sims:
            return {"available": False, "reason": "empty reference set"}

        order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        top = order[: max(1, k)]
        max_sim = round(sims[order[0]], 4)
        knn_mean = round(mean(sims[i] for i in top), 4)

        heavy = mol.GetNumHeavyAtoms()
        small_molecule_caveat = heavy < _SMALL_MOL_HEAVY_ATOM_FLOOR

        if max_sim >= _REF.t_in_domain:
            verdict, label = "in_domain", "In domain"
        elif max_sim >= _REF.t_borderline:
            verdict, label = "borderline", "Borderline"
        else:
            verdict, label = "extrapolation", "Extrapolation"

        # A very small molecule cannot earn an "in domain" verdict on ECFP4 similarity:
        # with few set bits, fragment overlap with something unrelated is easy. Ethanol
        # scored 0.278 against the seed set purely on generic aliphatic fragments.
        # Downgrade rather than assert confidence we do not have.
        if small_molecule_caveat and verdict == "in_domain":
            verdict, label = "borderline", "Borderline (small molecule)"

        interpretation = {
            "in_domain": "Structurally comparable to approved drugs; rule-based filters "
                         "and drug-like QSAR models are operating within their parameterisation.",
            "borderline": "Structurally atypical relative to approved drugs. Predictions "
                          "remain plausible but carry inflated uncertainty.",
            "extrapolation": "Structurally unlike anything in the reference set. Rule-based "
                             "filters and QSAR predictions are extrapolations and should not "
                             "be treated as quantitative.",
        }[verdict]

        out = {
            "available": True,
            "verdict": verdict,
            "label": label,
            "max_tanimoto": max_sim,
            "nearest_neighbour": _REF.names[order[0]],
            f"knn{len(top)}_mean_tanimoto": knn_mean,
            "neighbours": [
                {"name": _REF.names[i], "tanimoto": round(sims[i], 4)} for i in top
            ],
            "percentile_vs_reference": _REF.percentile_rank(max_sim),
            "thresholds": {
                "in_domain_at_or_above": _REF.t_in_domain,
                "extrapolation_below": _REF.t_borderline,
                "floor_is_binding": _REF.floor_is_binding,
            },
            "reference_set_size": _REF.size,
            "reference_provenance": _REF.provenance,
            "fingerprint": FP_SPEC,
            "interpretation": interpretation,
            "method": (
                "Max Tanimoto (ECFP4) to a reference set of approved small-molecule drugs. "
                "Thresholds calibrated as the 25th and 5th percentiles of the leave-one-out "
                "nearest-neighbour similarity distribution within that reference set."
            ),
            "caveat": (
                "Applicability domain is model-specific and structural. It indicates whether "
                "comparable chemistry was represented during parameterisation; it does not "
                "validate any individual prediction, and it does not apply to LLM-generated "
                "values, which have no defined training distribution."
            ),
        }

        if small_molecule_caveat:
            out["small_molecule_warning"] = (
                f"Only {heavy} heavy atoms. ECFP4 similarity is systematically depressed for "
                "very small structures, so this AD label is unreliable — interpret with caution."
            )
        return out

    except Exception as e:  # never let AD break a descriptor request
        return {"available": False, "reason": str(e)}
