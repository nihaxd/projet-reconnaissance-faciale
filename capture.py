import cv2
import numpy as np
import csv
import os

from features import preprocess_face, extract_features, draw_face_box
from config import CSV_PATH, DATA_DIR, MIN_CAPTURES

os.makedirs(DATA_DIR, exist_ok=True)


def save_features(name, features, csv_path=CSV_PATH):
    """Ajoute une ligne dans dataset.csv : nom;f1;f2;...;f30"""
    row = [name] + features.tolist()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(row)


def capture_person(name):
    """
    Lance la capture webcam pour une personne donnée.
    Retourne le nombre d'images capturées.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur : webcam non accessible.")
        return 0

    print(f"\n[CAPTURE] Personne : {name}")
    print(f"Appuyez sur 'c' pour capturer, 'q' pour terminer.")
    print(f"Objectif : {MIN_CAPTURES} captures (face neutre, expressions, rotations, éclairages).\n")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur : impossible de lire le flux vidéo.")
            break

        color = (0, 200, 0) if count >= MIN_CAPTURES else (0, 165, 255)
        label = f"{name}  [{count}/{MIN_CAPTURES}]"

        # Cadre de détection en temps réel
        display = draw_face_box(frame, color=color)
        cv2.putText(display, label, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        cv2.imshow("Capture - 'c' capturer | 'q' terminer", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            face = preprocess_face(frame)
            if face is None:
                print("  Aucun visage détecté, réessayez.")
                continue
            features = extract_features(face)
            save_features(name, features)
            count += 1
            print(f"  Capture {count} enregistrée.")

    cap.release()
    cv2.destroyAllWindows()

    if count < MIN_CAPTURES:
        print(f"\n[AVERTISSEMENT] Seulement {count}/{MIN_CAPTURES} captures pour {name}.")
    else:
        print(f"\n[OK] {count} captures enregistrées pour {name}.")
    return count


def main():
    name = input("Entrez le nom de la personne : ").strip()
    if not name:
        print("Nom vide, arrêt.")
        return
    capture_person(name)


if __name__ == "__main__":
    main()