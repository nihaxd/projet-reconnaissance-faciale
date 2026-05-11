import numpy as np
import pandas as pd
import os
from config import CSV_PATH

def load_dataset(csv_path=CSV_PATH):
    if not os.path.exists(csv_path):
        print("❌ dataset.csv introuvable. Lance capture.py d'abord.")
        return None, None
    df = pd.read_csv(csv_path, sep=';', header=None)
    names = df.iloc[:, 0].values
    feats = df.iloc[:, 1:].values.astype(np.float32)
    return names, feats


def compute_threshold(names, feats):
    persons = np.unique(names)

    intra_dists = []   # distances entre captures de la MÊME personne
    inter_dists = []   # distances entre captures de PERSONNES DIFFÉRENTES

    print("\n=== Distances INTRA-classe (même personne) ===")
    for person in persons:
        idx = np.where(names == person)[0]
        vecs = feats[idx]
        dists = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                d = float(np.linalg.norm(vecs[i] - vecs[j]))
                dists.append(d)
                intra_dists.append(d)
        if dists:
            print(f"  {person:20s}  n={len(vecs):3d}  "
                  f"mean={np.mean(dists):.4f}  "
                  f"max={np.max(dists):.4f}  "
                  f"min={np.min(dists):.4f}")
        else:
            print(f"  {person:20s}  ⚠️  une seule capture, pas de distance intra.")

    print("\n=== Distances INTER-classe (personnes différentes) ===")
    for i, p1 in enumerate(persons):
        for p2 in persons[i + 1:]:
            idx1 = np.where(names == p1)[0]
            idx2 = np.where(names == p2)[0]
            dists = []
            for v1 in feats[idx1]:
                for v2 in feats[idx2]:
                    d = float(np.linalg.norm(v1 - v2))
                    dists.append(d)
                    inter_dists.append(d)
            print(f"  {p1:15s} vs {p2:15s}  "
                  f"mean={np.mean(dists):.4f}  "
                  f"min={np.min(dists):.4f}")

    print("\n=== Résumé ===")
    if not intra_dists or not inter_dists:
        print("❌ Pas assez de données pour calculer un seuil.")
        return

    max_intra = max(intra_dists)
    min_inter = min(inter_dists)
    seuil_optimal = (max_intra + min_inter) / 2.0

    print(f"  Max intra  : {max_intra:.4f}")
    print(f"  Min inter  : {min_inter:.4f}")

    if min_inter > max_intra:
        print(f"  ✅ Bonne séparation ! Seuil optimal : {seuil_optimal:.4f}")
    else:
        print(f"  ⚠️  Chevauchement détecté (min_inter < max_intra)")
        print(f"     Seuil optimal (milieu) : {seuil_optimal:.4f}")
        print(f"     → Recapture recommandée ou ajustement des paramètres Snake.")

    print(f"\n  👉 Mets à jour config.py :")
    print(f"     THRESHOLD = {seuil_optimal:.2f}")

    # Estimation du taux d'erreur pour différents seuils candidats
    print("\n=== Simulation de différents seuils ===")
    seuils_candidats = np.arange(
        round(min(intra_dists + inter_dists), 2),
        round(max(intra_dists + inter_dists), 2),
        0.05
    )

    best_seuil = seuil_optimal
    best_err   = float('inf')

    for s in seuils_candidats:
        # FP = inter < seuil (inconnu accepté comme connu)
        fp = sum(1 for d in inter_dists if d < s)
        # FN = intra >= seuil (connu rejeté)
        fn = sum(1 for d in intra_dists if d >= s)
        err = (fp + fn) / (len(inter_dists) + len(intra_dists))
        if err < best_err:
            best_err   = err
            best_seuil = s

    print(f"  Seuil minimisant les erreurs : {best_seuil:.4f}  "
          f"(taux d'erreur estimé : {best_err*100:.1f}%)")
    print(f"\n  👉 Seuil recommandé final : THRESHOLD = {best_seuil:.2f}")


def main():
    names, feats = load_dataset()
    if names is None:
        return
    print(f"Dataset chargé : {len(names)} vecteurs, "
          f"{len(np.unique(names))} personnes.")
    compute_threshold(names, feats)


if __name__ == "__main__":
    main()