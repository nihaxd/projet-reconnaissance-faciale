import cv2
import numpy as np
import pandas as pd
import os
import threading
import tkinter as tk
from tkinter import simpledialog, messagebox

from features import preprocess_face, extract_features, draw_face_box
from config import CSV_PATH, THRESHOLD, DATA_DIR
from capture import capture_person

os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset(csv_path=CSV_PATH):
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path, sep=';', header=None)
    names = df.iloc[:, 0].values
    feats = df.iloc[:, 1:].values.astype(np.float32)
    return names, feats


def identify(query_features, names, feats, threshold=THRESHOLD):
    dists = np.linalg.norm(feats - query_features, axis=1)
    idx_sorted = np.argsort(dists)
    best_idx  = idx_sorted[0]
    best_dist = float(dists[best_idx])
    top3 = [(names[i], float(dists[i])) for i in idx_sorted[:3]]

    if best_dist >= threshold:
        return "Inconnu", best_dist, 0.0, top3
    conf = max(0.0, 1.0 - best_dist / threshold)
    return names[best_idx], best_dist, conf, top3


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class DoorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Porte securisee")
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(root, width=500, height=380, bg="#D2E2EC")
        self.canvas.pack()

        self.door_state  = "closed"
        self.light_color = "red"
        self.status_text = "En attente..."

        # Webcam partagee
        self.cap        = None
        self.last_frame = None
        self.frame_lock = threading.Lock()
        self.stop_cam   = False

        # Dataset
        self.names = None
        self.feats = None
        self._reload_dataset()

        self.draw_scene()
        self._start_camera()

        self.root.bind("<Key>", self.on_key)
        self.root.focus_force()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------

    def _reload_dataset(self):
        self.names, self.feats = load_dataset()
        if self.names is None:
            self.status_text = "Dataset manquant - appuyez sur E"
        else:
            self.status_text = "En attente d'identification"

    # -----------------------------------------------------------------------
    # Webcam
    # -----------------------------------------------------------------------

    def _start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_text = "Webcam non accessible"
            self.draw_scene()
            return
        t = threading.Thread(target=self._camera_loop, daemon=True)
        t.start()

    def _camera_loop(self):
        while not self.stop_cam:
            ret, frame = self.cap.read()
            if not ret:
                break
            with self.frame_lock:
                self.last_frame = frame
            display = draw_face_box(frame)
            cv2.putText(display, "Appuyez sur I dans la fenetre principale",
                        (8, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.imshow("Webcam", display)
            cv2.waitKey(1)

        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def _get_frame(self):
        with self.frame_lock:
            if self.last_frame is None:
                return None
            return self.last_frame.copy()

    # -----------------------------------------------------------------------
    # Dessin
    # -----------------------------------------------------------------------

    def draw_scene(self):
        self.canvas.delete("all")

        # Fond
        self.canvas.create_rectangle(0, 0, 500, 380, fill="#D2E2EC", outline="#D2E2EC")

        # Mur
        self.canvas.create_rectangle(80, 60, 320, 360, fill="#B0BEC5", outline="#90A4AE", width=2)

        # Porte
        door_color = "#98AA9D" if self.door_state == "open" else "#2D3536"
        self.canvas.create_rectangle(110, 80, 290, 355, fill=door_color, outline="#455A64", width=3)
        # Panneaux decoratifs
        self.canvas.create_rectangle(125, 95,  275, 190, fill="", outline="#546E7A", width=1)
        self.canvas.create_rectangle(125, 205, 275, 340, fill="", outline="#546E7A", width=1)
        # Poignee
        self.canvas.create_oval(270, 210, 284, 224, fill="#F2EFE2", outline="#333", width=1)

        # Voyant
        self.canvas.create_oval(390, 40, 430, 80, fill=self.light_color, outline="white", width=3)
        voyant_text  = "ACCES" if self.light_color == "green" else "REFUS"
        voyant_color = "#15803D" if self.light_color == "green" else "#B91C1C"
        self.canvas.create_text(410, 92, text=voyant_text, fill=voyant_color,
                                font=("Segoe UI", 9, "bold"))

        # Statut
        self.canvas.create_text(250, 22, text=self.status_text,
                                fill="#1F2937", font=("Segoe UI", 12, "bold"), anchor="center")

        # Bouton I - Identifier
        self.canvas.create_rectangle(345, 120, 480, 165, fill="#1D4ED8",
                                     outline="#1E40AF", width=2, tags="btn_id")
        self.canvas.create_text(412, 142, text="[I]  Identifier",
                                fill="white", font=("Segoe UI", 11, "bold"), tags="btn_id")

        # Bouton E - Enregistrer
        self.canvas.create_rectangle(345, 185, 480, 230, fill="#065F46",
                                     outline="#064E3B", width=2, tags="btn_reg")
        self.canvas.create_text(412, 207, text="[E]  Enregistrer",
                                fill="white", font=("Segoe UI", 11, "bold"), tags="btn_reg")

        # Bouton Q - Quitter
        self.canvas.create_rectangle(345, 250, 480, 295, fill="#7F1D1D",
                                     outline="#6B1A1A", width=2, tags="btn_quit")
        self.canvas.create_text(412, 272, text="[Q]  Quitter",
                                fill="white", font=("Segoe UI", 11, "bold"), tags="btn_quit")

        # Clic souris sur boutons
        self.canvas.tag_bind("btn_id",   "<Button-1>", lambda e: self._launch_identification())
        self.canvas.tag_bind("btn_reg",  "<Button-1>", lambda e: self._do_registration())
        self.canvas.tag_bind("btn_quit", "<Button-1>", lambda e: self.on_close())

        # Aide
        self.canvas.create_text(250, 370, text="Cliquez sur un bouton ou utilisez les touches I / E / Q",
                                fill="#6B7280", font=("Segoe UI", 8))

    # -----------------------------------------------------------------------
    # Clavier
    # -----------------------------------------------------------------------

    def on_key(self, event):
        key = event.keysym.lower()
        if key == 'q':
            self.on_close()
        elif key == 'i':
            self._launch_identification()
        elif key == 'e':
            self._do_registration()

    def on_close(self):
        self.stop_cam = True
        self.root.quit()

    # -----------------------------------------------------------------------
    # Identification
    # -----------------------------------------------------------------------

    def _launch_identification(self):
        t = threading.Thread(target=self._do_identification, daemon=True)
        t.start()

    def _do_identification(self):
        if self.names is None:
            self._set_status("Dataset manquant !", "red", "closed")
            return

        frame = self._get_frame()
        if frame is None:
            self._set_status("Webcam non prete", "red", "closed")
            return

        self._set_status("Identification en cours...", "red", "closed")

        face = preprocess_face(frame)
        if face is None:
            self._set_status("Aucun visage detecte", "red", "closed")
            return

        query_features = extract_features(face)
        name_pred, dist_best, conf, top3 = identify(
            query_features, self.names, self.feats
        )

        print(f"\n[ID] {name_pred}  dist={dist_best:.4f}  conf={conf*100:.1f}%")
        print("[ID] Top 3 :")
        for n, d in top3:
            print(f"       {n} : {d:.4f}")

        # Overlay webcam
        color_cv = (0, 200, 0) if name_pred != "Inconnu" else (0, 0, 220)
        label    = f"{name_pred}  dist={dist_best:.3f}  conf={conf*100:.0f}%"
        overlay  = frame.copy()
        cv2.rectangle(overlay, (4, 4), (frame.shape[1]-4, 50), color_cv, 2)
        cv2.putText(overlay, label, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_cv, 2)
        cv2.imshow("Webcam", overlay)
        cv2.waitKey(2000)

        # Mise a jour UI
        if name_pred == "Inconnu":
            self._set_status(f"ACCES REFUSE  (dist={dist_best:.3f})", "red", "closed")
        else:
            self._set_status(f"ACCES AUTORISE : {name_pred}  ({conf*100:.0f}%)", "green", "open")

    def _set_status(self, text, light, door):
        def _update():
            self.status_text = text
            self.light_color = light
            self.door_state  = door
            self.draw_scene()
        self.root.after(0, _update)

    # -----------------------------------------------------------------------
    # Enregistrement
    # -----------------------------------------------------------------------

    def _do_registration(self):
        name = simpledialog.askstring(
            "Enregistrement",
            "Nom de la personne a enregistrer :",
            parent=self.root
        )
        if not name or not name.strip():
            return

        name = name.strip()
        self._set_status(f"Capture en cours pour {name}...", "red", "closed")

        def _run():
            count = capture_person(name)
            self._reload_dataset()
            msg = (f"{count} captures enregistrees pour {name}."
                   if count > 0 else "Aucune capture effectuee.")
            self._set_status(msg, "red", "closed")
            messagebox.showinfo("Enregistrement", msg)

        threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()
    app  = DoorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()