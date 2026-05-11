import cv2
import numpy as np
import pandas as pd
import os
import csv

from features import preprocess_face, extract_features, draw_face_box
from config import CSV_PATH, TESTS_PATH, THRESHOLD, DATA_DIR

os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset(csv_path=CSV_PATH):
    if not os.path.exists(csv_path):
        print("Aucun dataset trouvé. Lance d'abord capture.py.")
        return None, None
    df = pd.read_csv(csv_path, sep=';', header=None)
    names = df.iloc[:, 0].values
    feats = df.iloc[:, 1:].values.astype(np.float32)
    return names, feats


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------

def identify(query_features, names, feats, threshold=THRESHOLD):
    dists = np.linalg.norm(feats - query_features, axis=1)
    idx_sorted = np.argsort(dists)
    best_idx  = idx_sorted[0]
    best_name = names[best_idx]
    best_dist = float(dists[best_idx])

    top3 = [(names[i], float(dists[i])) for i in idx_sorted[:3]]

    if best_dist >= threshold:
        confidence = 0.0
        name_display = "Inconnu"
    else:
        confidence   = max(0.0, 1.0 - best_dist / threshold)
        name_display = best_name

    return name_display, best_dist, confidence, top3


# ---------------------------------------------------------------------------
# Sauvegarde résultats de test
# ---------------------------------------------------------------------------

def save_test_result(true_name, predicted_name, csv_path=TESTS_PATH):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow([true_name, predicted_name])


# ---------------------------------------------------------------------------
# Boucle principale — une seule instance VideoCapture partagée
# ---------------------------------------------------------------------------

def main():
    names, feats = load_dataset()
    if names is None:
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur : webcam non accessible.")
        return

    print("Appuyez sur 'i' pour identifier | 'q' pour quitter")

    # On garde la dernière frame lue pour l'identification
    last_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur flux vidéo.")
            break

        last_frame = frame.copy()

        # Cadre de détection en temps réel
        display = draw_face_box(frame)
        cv2.putText(display, "I: identifier  Q: quitter", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.imshow("Identification - Porte securisee", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('i') and last_frame is not None:
            print("\n[ID] Tentative d'identification...")
            face = preprocess_face(last_frame)

            if face is None:
                print("[ID] Aucun visage détecté.")
                cv2.putText(display, "Aucun visage !", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Identification - Porte securisee", display)
                cv2.waitKey(800)
                continue

            query_features = extract_features(face)
            name_pred, dist_best, conf, top3 = identify(
                query_features, names, feats
            )

            # Affichage résultat
            color = (0, 200, 0) if name_pred != "Inconnu" else (0, 0, 255)
            label = f"{name_pred}  d={dist_best:.3f}  conf={conf*100:.0f}%"

            print(f"[ID] {label}")
            print("[ID] Top 3 :")
            for n, d in top3:
                print(f"     {n} : {d:.4f}")

            result_frame = last_frame.copy()
            cv2.putText(result_frame, label, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(result_frame, (5, 5), (500, 50), color, 2)

            cv2.imshow("Identification - Porte securisee", result_frame)
            cv2.waitKey(1500)

            # Demande du vrai nom pour évaluation
            print("[TEST] Entrez le vrai nom (Entrée = Inconnu) :")
            true_name = input().strip() or "Inconnu"
            save_test_result(true_name, name_pred)
            print(f"[TEST] Enregistré : vrai={true_name}, prédit={name_pred}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()