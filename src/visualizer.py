"""
Visualizer — Terminal Dashboard & Display
==========================================
Provides rich terminal-based visualization for the 5-class
speech classifier, including score bars, feature summaries,
and rule activation traces.

Reference: Phase 4, Section 10 of the Action Plan.
"""

from src.fuzzy_engine import CLASSES_5, generate_rule_trace


# ============================================================
# TERMINAL DISPLAY
# ============================================================

def format_score_bar(score, width=30):
    """Create a visual bar representation of a score value (0.0–1.0)."""
    filled = int(score * width)
    empty = width - filled
    return "█" * filled + "░" * empty


def display_classification_result(class_idx, confidence, details, features):
    """
    Display a rich terminal-based classification result.

    Matches the visualization spec from Phase 4 Section 10:
    ╔══════════════════════════════════════════════════════╗
    ║  🎤 Speech Classification Result                     ║
    ╠══════════════════════════════════════════════════════╣
    ║  Class: Confident Articulation     Confidence: 84%   ║
    ╠══════════════════════════════════════════════════════╣
    ║  Silence    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.02   ║
    ║  Confident  ████████████████████████░░░░░░░  0.72   ║
    ║  Hesitant   ███░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.10   ║
    ║  Anxious    █████░░░░░░░░░░░░░░░░░░░░░░░░░  0.14   ║
    ║  Monotone   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.02   ║
    ╚══════════════════════════════════════════════════════╝
    """
    class_name = CLASSES_5[class_idx]
    scores = details.get("scores", {})
    is_ambiguous = details.get("is_ambiguous", False)
    margin = details.get("margin", 0)

    # Header
    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║  🎤 Speech Classification Result                     ║")
    print("  ╠══════════════════════════════════════════════════════╣")

    # Class & confidence
    cls_display = f"{class_name:<30}"
    conf_display = f"Confidence: {confidence}%"
    print(f"  ║  Class: {cls_display} {conf_display:<12}║")

    if is_ambiguous:
        print(f"  ║  ⚠️  AMBIGUOUS (margin: {margin:.3f})                     ║")

    print("  ╠══════════════════════════════════════════════════════╣")

    # Score bars
    short_names = ["Silence  ", "Confident", "Hesitant ", "Anxious  ", "Monotone "]
    for i, cls in enumerate(CLASSES_5):
        score = scores.get(cls, 0)
        bar = format_score_bar(score)
        print(f"  ║  {short_names[i]}  {bar}  {score:.2f}   ║")

    print("  ╠══════════════════════════════════════════════════════╣")

    # Feature summary
    avg = features.get("avg_level", 0)
    ratio = features.get("speech_ratio", 0)
    gaps = features.get("num_gaps", 0)
    curv = features.get("prosody_curvature", 0)
    srv = features.get("speech_rate_variability", 0)
    pde = features.get("pause_duration_entropy", 0)
    drift = features.get("intensity_drift", 0)
    latency = features.get("response_latency", 0)

    print(f"  ║  Volume={avg:<6.1f}  Speech={ratio:<5.2f}  Pauses={gaps:<3}  Melody={curv:<5.2f}║")
    print(f"  ║  Rhythm={srv:<6.2f}  PauseVar={pde:<5.2f}  Trend={drift:<+6.3f}       ║")
    print(f"  ║  Reaction={latency:<6.0f}ms                                   ║")
    print("  ╚══════════════════════════════════════════════════════╝")


def display_rule_trace(class_idx, confidence, details):
    """Display the full rule activation trace for explainability."""
    trace = generate_rule_trace(class_idx, confidence, details)
    print()
    print("  ┌─ Rule Activation Trace ──────────────────────────────")
    for line in trace.split("\n"):
        print(f"  │ {line}")
    print("  └─────────────────────────────────────────────────────")


def display_features_summary(features):
    """Display a concise summary of extracted features."""
    print(f"  Features:")
    print(f"    avg_level:               {features.get('avg_level', 0):.1f}")
    print(f"    speech_ratio:            {features.get('speech_ratio', 0):.3f}")
    print(f"    num_gaps:                {features.get('num_gaps', 0)}")
    print(f"    prosody_curvature:       {features.get('prosody_curvature', 0):.3f}")
    print(f"    speech_rate_variability: {features.get('speech_rate_variability', 0):.3f}")
    print(f"    pause_duration_entropy:  {features.get('pause_duration_entropy', 0):.3f}")
    print(f"    intensity_drift:         {features.get('intensity_drift', 0):+.4f}")
    print(f"    response_latency:        {features.get('response_latency', 0):.0f} ms")


def display_training_summary(accuracy, confusion, n_classes=5):
    """Display a formatted training/validation summary."""
    classes = CLASSES_5[:n_classes]
    n_total = sum(sum(row) for row in confusion)
    n_correct = sum(confusion[i][i] for i in range(n_classes))

    # Standardized Box Width (64 inner characters)
    box_width = 64
    indent = "  "

    print()
    print(f"{indent}╔{'═' * box_width}╗")
    print(f"{indent}║{'📊 Training Validation Results':^64}║")
    print(f"{indent}╠{'═' * box_width}╣")
    
    accuracy_text = f"Accuracy: {accuracy * 100:.1f}% ({n_correct}/{n_total})"
    print(f"{indent}║  {accuracy_text:<62}║")
    print(f"{indent}╠{'═' * box_width}╣")

    # Confusion Matrix Title
    print(f"{indent}║{'Confusion Matrix':^64}║")
    
    # Confusion Matrix Header
    # Class columns: 9 chars each. Total for 5 classes = 45. 
    # Row label: 17 chars. Margin: 2 chars.
    header = f"{indent}║  {'':<17}"
    for cls in classes:
        # Improved spelling/truncation for header
        name = cls.replace("Articulation", "Art.").replace("Disfluency", "Disf.").replace("Urgency", "Urg.").replace("Monotone", "Mono.")
        header += f"{name[:7]:>9}"
    header += f"{'':>2}║"
    print(header)

    # Confusion Matrix Rows
    for i, cls in enumerate(classes):
        row = f"{indent}║  {cls[:17]:<17}"
        for j in range(n_classes):
            row += f"{confusion[i][j]:>9}"
        row += f"{'':>2}║"
        print(row)

    print(f"{indent}╠{'═' * box_width}╣")

    # Per-Class Metrics Title
    print(f"{indent}║{'Per-Class Metrics':^64}║")
    
    # Metric Rows
    for i, cls in enumerate(classes):
        tp = confusion[i][i]
        fp = sum(confusion[j][i] for j in range(n_classes)) - tp
        fn = sum(confusion[i][j] for j in range(n_classes)) - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        # Consistent spacing for metric values
        metrics = f"P={p:.2f}   R={r:.2f}   F1={f1:.2f}"
        print(f"{indent}║  {cls[:20]:<25}{metrics:<37}║")

    print(f"{indent}╚{'═' * box_width}╝")
