import csv
import os
import numpy as np

from config import TESTS_PATH, REPORTS_DIR

os.makedirs(REPORTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_tests(csv_path=TESTS_PATH):
    if not os.path.exists(csv_path):
        print("Aucun fichier tests_results.csv. Lance recognize.py d'abord.")
        return []
    tests = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter=';'):
            if len(row) == 2:
                tests.append((row[0].strip(), row[1].strip()))
    return tests


# ---------------------------------------------------------------------------
# Métriques binaires (connu / inconnu)
# ---------------------------------------------------------------------------

def compute_metrics(tests):
    TP = FP = TN = FN = 0
    for vrai, predit in tests:
        vrai_connu   = vrai   != "Inconnu"
        predit_connu = predit != "Inconnu"

        if vrai_connu:
            if predit == vrai:    TP += 1
            elif predit_connu:    FP += 1   # mauvaise identité acceptée
            else:                 FN += 1   # connu rejeté
        else:
            if not predit_connu:  TN += 1
            else:                 FP += 1   # inconnu accepté

    total     = TP + FP + TN + FN
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rappel    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specif    = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    erreur    = (FP + FN) / total if total > 0 else 0.0
    f1        = (2 * precision * rappel / (precision + rappel)
                 if (precision + rappel) > 0 else 0.0)

    return dict(TP=TP, FP=FP, TN=TN, FN=FN,
                precision=precision, rappel=rappel,
                specificite=specif, erreur=erreur, f1=f1,
                total=total)


# ---------------------------------------------------------------------------
# Matrice de confusion multi-classes (toutes les personnes + Inconnu)
# ---------------------------------------------------------------------------

def build_confusion_matrix(tests):
    """
    Construit la matrice de confusion multi-classes.
    Retourne (matrix, labels) avec labels triés alphabétiquement.
    """
    all_labels = sorted(set(v for v, _ in tests) | set(p for _, p in tests))
    idx = {l: i for i, l in enumerate(all_labels)}
    n = len(all_labels)
    matrix = np.zeros((n, n), dtype=int)

    for vrai, predit in tests:
        matrix[idx[vrai], idx[predit]] += 1

    return matrix, all_labels


# ---------------------------------------------------------------------------
# Export PNG avec matplotlib
# ---------------------------------------------------------------------------

def save_confusion_matrix_png(matrix, labels, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("[WARN] matplotlib non installé, pas de graphique généré.")
        return

    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))

    im = ax.imshow(matrix, cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_xlabel("Prédit", fontsize=11)
    ax.set_ylabel("Réel", fontsize=11)
    ax.set_title("Matrice de confusion", fontsize=13, pad=14)

    # Valeurs dans les cases
    thresh = matrix.max() / 2.0
    for i in range(n):
        for j in range(n):
            color = "white" if matrix[i, j] > thresh else "black"
            ax.text(j, i, str(matrix[i, j]),
                    ha='center', va='center', fontsize=10, color=color,
                    fontweight='bold' if i == j else 'normal')

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[OK] Matrice de confusion exportée : {path}")


def save_metrics_txt(metrics, path):
    lines = [
        "=== Résultats de l'évaluation ===\n",
        f"Total tests : {metrics['total']}\n",
        f"TP={metrics['TP']}  FP={metrics['FP']}  TN={metrics['TN']}  FN={metrics['FN']}\n",
        "\n",
        f"Précision    : {metrics['precision']*100:.1f} %\n",
        f"Rappel       : {metrics['rappel']*100:.1f} %\n",
        f"Spécificité  : {metrics['specificite']*100:.1f} %\n",
        f"F1-score     : {metrics['f1']*100:.1f} %\n",
        f"Taux d'erreur: {metrics['erreur']*100:.1f} %\n",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[OK] Métriques exportées : {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tests = load_tests()
    if not tests:
        return

    print(f"\n{len(tests)} test(s) chargé(s) :\n")
    for i, (v, p) in enumerate(tests, 1):
        status = "✓" if v == p else "✗"
        print(f"  {i:02d}. [{status}] Réel={v:<20} Prédit={p}")

    # Métriques binaires
    metrics = compute_metrics(tests)
    print("\n=== Métriques ===")
    print(f"TP={metrics['TP']}  FP={metrics['FP']}  TN={metrics['TN']}  FN={metrics['FN']}")
    print(f"Précision    : {metrics['precision']*100:.1f} %")
    print(f"Rappel       : {metrics['rappel']*100:.1f} %")
    print(f"Spécificité  : {metrics['specificite']*100:.1f} %")
    print(f"F1-score     : {metrics['f1']*100:.1f} %")
    print(f"Taux d'erreur: {metrics['erreur']*100:.1f} %")

    # Matrice de confusion multi-classes
    matrix, labels = build_confusion_matrix(tests)
    print("\n=== Matrice de confusion ===")
    header = " " * 14 + "  ".join(f"{l[:8]:>8}" for l in labels)
    print(header)
    for i, label in enumerate(labels):
        row = "  ".join(f"{matrix[i,j]:>8}" for j in range(len(labels)))
        print(f"{label[:14]:>14}  {row}")

    # Export
    png_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    txt_path = os.path.join(REPORTS_DIR, "metrics.txt")
    save_confusion_matrix_png(matrix, labels, png_path)
    save_metrics_txt(metrics, txt_path)


if __name__ == "__main__":
    main()