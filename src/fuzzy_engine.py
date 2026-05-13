"""
Fuzzy Inference Engine — 5-Class Mamdani System
=================================================
Implements membership functions, 13-rule evaluation, and
defuzzification for the 5-class speech classifier.

Classes:
  1. Ambient Silence
  2. Confident Articulation
  3. Hesitant Disfluency
  4. Anxious Urgency
  5. Disengaged Monotone

Reference: Phase 3 & Phase 4 of the Action Plan.
"""

import math


# ============================================================
# OUTPUT CLASS DEFINITIONS
# ============================================================

CLASSES_5 = [
    "Ambient Silence",
    "Confident Articulation",
    "Hesitant Disfluency",
    "Anxious Urgency",
    "Disengaged Monotone",
]

CLASS_INDICES_5 = {name: i for i, name in enumerate(CLASSES_5)}

RESULT_TAGS_5 = ["SILENCE", "CONFIDENT", "HESITANT", "ANXIOUS", "MONOTONE"]




# ============================================================
# MEMBERSHIP FUNCTIONS
# ============================================================

def left_shoulder(x, a, b):
    """Full membership below a, linear decline to 0 at b."""
    if x <= a:
        return 1.0
    elif x >= b:
        return 0.0
    else:
        return (b - x) / (b - a)


def right_shoulder(x, a, b):
    """Zero below a, linear rise to full membership at b."""
    if x <= a:
        return 0.0
    elif x >= b:
        return 1.0
    else:
        return (x - a) / (b - a)


def triangular(x, a, b, c):
    """Zero at a and c, peak of 1.0 at b."""
    if x <= a or x >= c:
        return 0.0
    elif x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    else:
        return (c - x) / (c - b) if c != b else 1.0


def gaussian(x, mu, sigma):
    """Gaussian bell curve centered at mu."""
    if sigma == 0:
        return 1.0 if x == mu else 0.0
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def sigmoid_mf(x, center, steepness):
    """S-shaped curve transitioning at center."""
    exponent = -steepness * (x - center)
    if exponent > 500:
        return 0.0
    elif exponent < -500:
        return 1.0
    return 1.0 / (1.0 + math.exp(exponent))


# Legacy MF wrappers (3-parameter form used by trainer)
def tri_mf(x, a, b, c):
    """Triangular membership function → [0.0, 1.0] (legacy 3-param form)."""
    return triangular(x, a, b, c)


def left_mf(x, a, b, c):
    """Left shoulder membership function → [0.0, 1.0] (legacy 3-param form)."""
    if x <= b:
        return 1.0
    elif x >= c:
        return 0.0
    else:
        return (c - x) / (c - b) if c != b else 1.0


def right_mf(x, a, b, c):
    """Right shoulder membership function → [0.0, 1.0] (legacy 3-param form)."""
    if x <= a:
        return 0.0
    elif x >= b:
        return 1.0
    else:
        return (x - a) / (b - a) if b != a else 1.0


# ============================================================
# DEFAULT MF PARAMETERS — 5-CLASS SYSTEM
# ============================================================

# Each variable has 3 linguistic terms (LOW/MED/HIGH), stored as (a, b, c) tuples.
# For left_mf: full at b, zero at c; a is unused (convention).
# For tri_mf: zero at a, peak at b, zero at c.
# For right_mf: zero at a, full at b; c is unused (convention).

DEFAULT_PARAMS_5 = {
    # ── avg_level (0–255) ──
    "LEVEL_LOW":   (0, 0, 30),         # Ambient: very low energy
    "LEVEL_MED":   (15, 65, 130),       # Moderate: speech present but not projected
    "LEVEL_HIGH":  (80, 160, 255),      # High: projected voice

    # ── speech_ratio (0.0–1.0) ──
    "RATIO_LOW":   (0.0, 0.0, 0.15),   # Almost no above-threshold samples
    "RATIO_MED":   (0.08, 0.35, 0.65), # Substantial speech with gaps
    "RATIO_HIGH":  (0.50, 0.80, 1.0),  # Filling most of the window

    # ── num_gaps (0–25) ──
    "GAPS_FEW":    (0, 0, 4),           # Few pauses
    "GAPS_SOME":   (2, 6, 12),          # Moderate fragmentation
    "GAPS_MANY":   (8, 15, 25),         # Many speech-silence transitions

    # ── prosody_curvature (0–50+) ──
    "CURV_LOW":    (0, 0, 5),           # Flat amplitude envelope
    "CURV_MED":    (3, 10, 22),         # Moderate dynamics
    "CURV_HIGH":   (15, 30, 50),        # Highly dynamic speech
    # Gaussian variant for Confident detection (bell curve, less extreme)
    "CURV_HI_GAUSS_MU":  25.0,          # Center of Gaussian for curv_hi
    "CURV_HI_GAUSS_SIG": 10.0,          # Width (sigma) of Gaussian

    # ── speech_rate_variability (0–5+) ──
    "SRV_LOW":     (0, 0, 0.8),         # Consistent tempo
    "SRV_MED":     (0.5, 1.5, 2.5),    # Moderate variability
    "SRV_HIGH":    (1.5, 3.0, 5.0),    # Highly variable rate

    # ── pause_duration_entropy (0–2.0) ──
    "PDE_LOW":     (0, 0, 0.4),         # Regular, predictable pauses
    "PDE_MED":     (0.2, 0.8, 1.4),    # Moderate entropy
    "PDE_HIGH":    (0.8, 1.5, 2.0),    # Unpredictable pause lengths

    # ── intensity_drift (-2 to +2) ──
    "DRIFT_NEG":   (-2.0, -2.0, -0.1), # Trailing off (negative)
    "DRIFT_ZERO":  (-0.3, 0.0, 0.3),   # Stable
    "DRIFT_POS":   (0.1, 2.0, 2.0),    # Getting louder (positive)

    # ── response_latency (0–3000 ms) ──
    "LATENCY_LOW":  (0, 0, 400),        # Quick onset
    "LATENCY_MED":  (200, 800, 1500),   # Moderate delay
    "LATENCY_HIGH": (1000, 2500, 3000), # Slow to begin / no speech
}


# ============================================================
# FUZZIFICATION
# ============================================================

def fuzzify(features, params):
    """
    Compute all 24 membership values from 8 input features.

    Args:
        features: Dict with keys matching the 8 feature names.
        params: Dict of MF parameter tuples (a, b, c).

    Returns:
        Dict of 24 membership values keyed by linguistic term names.
    """
    avg = features.get("avg_level", 0)
    ratio = features.get("speech_ratio", 0)
    gaps = features.get("num_gaps", 0)
    curv = features.get("prosody_curvature", 0)
    srv = features.get("speech_rate_variability", 0)
    pde = features.get("pause_duration_entropy", 0)
    drift = features.get("intensity_drift", 0)
    latency = features.get("response_latency", 3000)

    m = {
        # avg_level
        "lvl_lo":  left_mf(avg, *params["LEVEL_LOW"]),
        "lvl_md":  tri_mf(avg, *params["LEVEL_MED"]),
        "lvl_hi":  right_mf(avg, *params["LEVEL_HIGH"]),

        # speech_ratio
        "rat_lo":  left_mf(ratio, *params["RATIO_LOW"]),
        "rat_md":  tri_mf(ratio, *params["RATIO_MED"]),
        "rat_hi":  right_mf(ratio, *params["RATIO_HIGH"]),

        # num_gaps
        "gap_fw":  left_mf(gaps, *params["GAPS_FEW"]),
        "gap_sm":  tri_mf(gaps, *params["GAPS_SOME"]),
        "gap_mn":  right_mf(gaps, *params["GAPS_MANY"]),

        # prosody_curvature
        "curv_lo": left_mf(curv, *params["CURV_LOW"]),
        "curv_md": tri_mf(curv, *params["CURV_MED"]),
        "curv_hi": right_mf(curv, *params["CURV_HIGH"]),
        # Gaussian variant for smoother Confident detection (Phase 3 spec)
        "curv_hi_gauss": gaussian(
            curv,
            params.get("CURV_HI_GAUSS_MU", 25.0),
            params.get("CURV_HI_GAUSS_SIG", 10.0),
        ),

        # speech_rate_variability
        "srv_lo":  left_mf(srv, *params["SRV_LOW"]),
        "srv_md":  tri_mf(srv, *params["SRV_MED"]),
        "srv_hi":  right_mf(srv, *params["SRV_HIGH"]),

        # pause_duration_entropy
        "pde_lo":  left_mf(pde, *params["PDE_LOW"]),
        "pde_md":  tri_mf(pde, *params["PDE_MED"]),
        "pde_hi":  right_mf(pde, *params["PDE_HIGH"]),

        # intensity_drift
        "drift_neg":  left_mf(drift, *params["DRIFT_NEG"]),
        "drift_zero": tri_mf(drift, *params["DRIFT_ZERO"]),
        "drift_pos":  right_mf(drift, *params["DRIFT_POS"]),

        # response_latency
        "latency_lo":  left_mf(latency, *params["LATENCY_LOW"]),
        "latency_md":  tri_mf(latency, *params["LATENCY_MED"]),
        "latency_hi":  right_mf(latency, *params["LATENCY_HIGH"]),
    }

    return m


# ============================================================
# RULE EVALUATION — 13-RULE SYSTEM
# ============================================================

def evaluate_rules(memberships, rule_weights=None):
    """
    Evaluate all 13 rules and return 5 class scores.

    Rules from Phase 3:
      R01: lvl_lo ∧ rat_lo ∧ latency_hi         → Ambient Silence
      R02: lvl_lo ∧ rat_lo ∧ curv_lo             → Ambient Silence
      R03: lvl_hi ∧ rat_hi ∧ gap_fw ∧ curv_hi    → Confident Articulation
      R04: lvl_md ∧ rat_hi ∧ srv_lo ∧ curv_md    → Confident Articulation
      R05: rat_md ∧ gap_mn ∧ pde_hi ∧ srv_hi     → Hesitant Disfluency
      R06: rat_md ∧ drift_neg ∧ latency_hi ∧ gap_mn → Hesitant Disfluency
      R07: gap_mn ∧ srv_hi ∧ curv_md             → Hesitant Disfluency
      R08: lvl_hi ∧ rat_hi ∧ curv_lo ∧ srv_md    → Anxious Urgency
      R09: lvl_hi ∧ gap_fw ∧ pde_lo ∧ drift_pos  → Anxious Urgency
      R10: rat_hi ∧ curv_lo ∧ srv_md              → Anxious Urgency
      R11: lvl_md ∧ curv_lo ∧ srv_lo ∧ pde_lo    → Disengaged Monotone
      R12: rat_md ∧ gap_sm ∧ curv_lo ∧ drift_zero → Disengaged Monotone
      R13: lvl_md ∧ rat_md ∧ srv_lo ∧ latency_md → Disengaged Monotone

    Args:
        memberships: Dict of 24 membership values from fuzzify().
        rule_weights: Optional list of 13 per-rule weights (default all 1.0).

    Returns:
        (scores, rules_dict):
          scores: list of 5 class scores
          rules_dict: dict mapping rule name to its firing strength
    """
    m = memberships
    if rule_weights is None:
        rule_weights = [1.0] * 13

    # Evaluate each rule (Mamdani min-AND)
    # Note: R06 originally specified vot_lo (voice onset timing), a Tier 2 variable
    # not implemented. Instead of duplicating gap_mn (used in R05/R07), we use pde_md
    # to capture irregular pause patterns, providing unique discriminative power.
    rules = {
        # Ambient Silence
        "R01": min(m["lvl_lo"], m["rat_lo"], m["latency_hi"]) * rule_weights[0],
        "R02": min(m["lvl_lo"], m["rat_lo"], m["curv_lo"]) * rule_weights[1],

        # Confident Articulation (R03 uses Gaussian MF for curvature per Phase 3)
        "R03": min(m["lvl_hi"], m["rat_hi"], m["gap_fw"], m["curv_hi_gauss"]) * rule_weights[2],
        "R04": min(m["lvl_md"], m["rat_hi"], m["srv_lo"], m["curv_md"]) * rule_weights[3],

        # Hesitant Disfluency
        "R05": min(m["rat_md"], m["gap_mn"], m["pde_hi"], m["srv_hi"]) * rule_weights[4],
        "R06": min(m["rat_md"], m["drift_neg"], m["latency_hi"], m["pde_md"]) * rule_weights[5],
        "R07": min(m["gap_mn"], m["srv_hi"], m["curv_md"]) * rule_weights[6],

        # Anxious Urgency
        "R08": min(m["lvl_hi"], m["rat_hi"], m["curv_lo"], m["srv_md"]) * rule_weights[7],
        "R09": min(m["lvl_hi"], m["gap_fw"], m["pde_lo"], m["drift_pos"]) * rule_weights[8],
        "R10": min(m["rat_hi"], m["curv_lo"], m["srv_md"]) * rule_weights[9],

        # Disengaged Monotone
        "R11": min(m["lvl_md"], m["curv_lo"], m["srv_lo"], m["pde_lo"]) * rule_weights[10],
        "R12": min(m["rat_md"], m["gap_sm"], m["curv_lo"], m["drift_zero"]) * rule_weights[11],
        "R13": min(m["lvl_md"], m["rat_md"], m["srv_lo"], m["latency_md"]) * rule_weights[12],
    }

    # Aggregate: max per class
    scores = [
        max(rules["R01"], rules["R02"]),                       # 0: Ambient Silence
        max(rules["R03"], rules["R04"]),                       # 1: Confident Articulation
        max(rules["R05"], rules["R06"], rules["R07"]),         # 2: Hesitant Disfluency
        max(rules["R08"], rules["R09"], rules["R10"]),         # 3: Anxious Urgency
        max(rules["R11"], rules["R12"], rules["R13"]),         # 4: Disengaged Monotone
    ]

    return scores, rules


# ============================================================
# DEFUZZIFICATION
# ============================================================

def defuzzify(scores):
    """
    Defuzzify 5 class scores into a classification result.

    Returns:
        (class_idx, confidence%, margin, is_ambiguous)
    """
    total = sum(scores)
    if total == 0:
        return 0, 0, 0.0, True

    winner_idx = max(range(len(scores)), key=lambda i: scores[i])
    confidence = int((scores[winner_idx] / total) * 100)

    # Ambiguity detection: margin between top-2 scores
    sorted_scores = sorted(scores, reverse=True)
    margin = sorted_scores[0] - sorted_scores[1]
    is_ambiguous = margin < 0.1

    return winner_idx, confidence, margin, is_ambiguous


# ============================================================
# COMPLETE CLASSIFICATION PIPELINE
# ============================================================

def fuzzy_classify_5(features, params, rule_weights=None):
    """
    Full 5-class fuzzy classification pipeline.

    Args:
        features: Dict of 8 feature values from extract_features_v2().
        params: Dict of MF parameter tuples.
        rule_weights: Optional per-rule weights (list of 13 floats).

    Returns:
        (class_idx, confidence%, details_dict)
    """
    # Step 1: Fuzzification
    memberships = fuzzify(features, params)

    # Step 2: Rule evaluation
    scores, rules = evaluate_rules(memberships, rule_weights)

    # Step 3: Defuzzification
    class_idx, confidence, margin, is_ambiguous = defuzzify(scores)

    # Build details dict for explainability
    details = {
        "scores": {CLASSES_5[i]: round(scores[i], 4) for i in range(5)},
        "rules": {name: round(val, 4) for name, val in rules.items()},
        "memberships": {name: round(val, 4) for name, val in memberships.items()},
        "confidence": confidence,
        "margin": round(margin, 4),
        "is_ambiguous": is_ambiguous,
    }

    return class_idx, confidence, details


def generate_rule_trace(class_idx, confidence, details):
    """
    Generate a human-readable rule activation trace for explainability.

    Returns:
        A multi-line string explaining the classification decision.
    """
    class_name = CLASSES_5[class_idx]
    lines = []
    lines.append(f"Classification: {class_name} (confidence: {confidence}%)")

    if details.get("is_ambiguous"):
        lines.append("⚠️  AMBIGUOUS — margin between top-2 scores < 0.1")

    lines.append("")

    # Identify which rules contributed to the winner
    # Map rules to classes
    rule_class_map = {
        "R01": 0, "R02": 0,
        "R03": 1, "R04": 1,
        "R05": 2, "R06": 2, "R07": 2,
        "R08": 3, "R09": 3, "R10": 3,
        "R11": 4, "R12": 4, "R13": 4,
    }

    winning_rules = []
    competing_rules = []

    for rule_name, strength in details["rules"].items():
        if strength > 0:
            target_class = rule_class_map[rule_name]
            if target_class == class_idx:
                winning_rules.append((rule_name, strength))
            else:
                competing_rules.append((rule_name, strength, CLASSES_5[target_class]))

    winning_rules.sort(key=lambda x: x[1], reverse=True)
    competing_rules.sort(key=lambda x: x[1], reverse=True)

    if winning_rules:
        lines.append("Winning rules:")
        for name, strength in winning_rules:
            lines.append(f"  {name} = {strength:.4f}")

    if competing_rules:
        lines.append("")
        lines.append("Competing rules:")
        for name, strength, cls_name in competing_rules[:3]:  # Show top 3
            lines.append(f"  {name} = {strength:.4f}  → {cls_name}")

    lines.append("")
    lines.append("DISCLAIMER: This is an acoustic analysis, not a psychological assessment.")

    return "\n".join(lines)


