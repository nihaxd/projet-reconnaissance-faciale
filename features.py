import cv2
import numpy as np
from skimage.segmentation import active_contour
from skimage.filters import gaussian

from config import (FACE_SIZE, SNAKE_ALPHA, SNAKE_BETA, SNAKE_GAMMA,
                    SNAKE_ITER, SNAKE_POINTS, FEAT_DIM)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)


# ---------------------------------------------------------------------------
# UTILITAIRE : dessiner le cadre de détection sur une frame BGR
# ---------------------------------------------------------------------------

def draw_face_box(frame, color=(0, 220, 0)):
    """
    Détecte les visages dans `frame` (BGR) et dessine :
      - Un rectangle coloré autour du visage
      - Un point sur chaque œil détecté
      - Un texte "Visage détecté" / "Aucun visage"
    Retourne la frame annotée (copie).
    """
    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        cv2.putText(display, "Aucun visage detecte", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 220), 2)
        return display

    # Plus grand visage uniquement
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Coin supérieur gauche stylisé (cadre en L)
    lw = 3          # épaisseur trait
    ls = 20         # longueur segment de coin
    # Coin haut-gauche
    cv2.line(display, (x, y),         (x + ls, y),      color, lw)
    cv2.line(display, (x, y),         (x, y + ls),      color, lw)
    # Coin haut-droit
    cv2.line(display, (x+w, y),       (x+w - ls, y),    color, lw)
    cv2.line(display, (x+w, y),       (x+w, y + ls),    color, lw)
    # Coin bas-gauche
    cv2.line(display, (x, y+h),       (x + ls, y+h),    color, lw)
    cv2.line(display, (x, y+h),       (x, y+h - ls),    color, lw)
    # Coin bas-droit
    cv2.line(display, (x+w, y+h),     (x+w - ls, y+h),  color, lw)
    cv2.line(display, (x+w, y+h),     (x+w, y+h - ls),  color, lw)

    # Détection des yeux dans la ROI
    face_gray = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 5)
    for (ex, ey, ew, eh) in eyes:
        cx_eye = x + ex + ew // 2
        cy_eye = y + ey + eh // 2
        cv2.circle(display, (cx_eye, cy_eye), 4, color, -1)
        cv2.circle(display, (cx_eye, cy_eye), 7, color,  1)

    # Label
    label = f"Visage detecte  {w}x{h}px"
    cv2.putText(display, label, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return display


# ---------------------------------------------------------------------------
# 1. PRÉTRAITEMENT
# ---------------------------------------------------------------------------

def preprocess_face(frame):
    """
    Détecte le plus grand visage, aligne par les yeux si possible,
    renvoie une image 128x128 niveaux de gris égalisée, ou None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_gray = gray[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 5)
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        (ex1, ey1, ew1, eh1) = eyes[0]
        (ex2, ey2, ew2, eh2) = eyes[1]

        eye1_center = (ex1 + ew1 / 2.0, ey1 + eh1 / 2.0)
        eye2_center = (ex2 + ew2 / 2.0, ey2 + eh2 / 2.0)

        if eye1_center[0] < eye2_center[0]:
            left_eye, right_eye = eye1_center, eye2_center
        else:
            left_eye, right_eye = eye2_center, eye1_center

        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        center_face = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center_face, angle, 1.0)
        face_aligned = cv2.warpAffine(face_gray, M, (w, h))
    else:
        face_aligned = face_gray

    face_aligned = cv2.resize(face_aligned, (FACE_SIZE, FACE_SIZE))
    face_aligned = cv2.equalizeHist(face_aligned)
    return face_aligned


# ---------------------------------------------------------------------------
# 2. CONTOUR ACTIF (SNAKE)
# ---------------------------------------------------------------------------

def extract_face_contour(face_img):
    """
    Applique un Snake sur l'image 128x128, retourne tableau (N, 2) [r, c].

    IMPORTANT : active_contour attend une image en float [0, 1].
    On normalise en float64 avant gaussian + Snake.
    """
    img_f = face_img.astype(np.float64) / 255.0
    img = gaussian(img_f, sigma=2, preserve_range=True)
    h, w = img.shape
    cy, cx = h / 2.0, w / 2.0
    radius = min(h, w) * 0.4

    s = np.linspace(0, 2 * np.pi, SNAKE_POINTS)
    r = cy + radius * np.sin(s)
    c = cx + radius * np.cos(s)
    init = np.array([r, c]).T

    snake = active_contour(
        img, init,
        alpha=SNAKE_ALPHA,
        beta=SNAKE_BETA,
        gamma=SNAKE_GAMMA,
        w_line=0,
        w_edge=1,
        max_num_iter=SNAKE_ITER,
        boundary_condition='periodic'
    )
    return snake


# ---------------------------------------------------------------------------
# 3. EXTRACTION DE FEATURES — VECTEUR 30D ENRICHI (sans padding dupliqué)
# ---------------------------------------------------------------------------
#
# Répartition :
#   [0-5]   Géométrie Snake          (6)
#   [6-13]  Intensités régions       (8)  mean+std × 4 régions
#   [14-16] Intensités globales      (3)  left_half, right_half, global
#   [17-20] Symétrie & gradient      (4)
#   [21-26] Histogramme global       (6)  6 bins normalisés
#   [27-29] Textures Sobel           (3)  mean_grad, std_grad, grad_asym
#   TOTAL = 30
# ---------------------------------------------------------------------------

def _region_stats(face_img, y0, y1, x0, x1):
    """Mean et std normalisés d'une ROI."""
    roi = face_img[y0:y1, x0:x1].astype(np.float32)
    return float(roi.mean()) / 255.0, float(roi.std()) / 255.0


def extract_features(face_img):
    """
    Extrait un vecteur 30D réel (aucun padding dupliqué).
    """
    h, w = face_img.shape  # 128 × 128

    # --- A. Géométrie Snake (6 features) ---
    snake = extract_face_contour(face_img)
    r, c = snake[:, 0], snake[:, 1]

    height_face = float(r.max() - r.min())
    width_face  = float(c.max() - c.min())
    ratio_wh    = width_face / height_face if height_face > 0 else 0.0
    diffs       = np.diff(snake, axis=0)
    perim       = float(np.sum(np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)))
    r_mean      = float(r.mean())
    c_mean      = float(c.mean())

    feats_snake = [
        height_face / h,
        width_face  / w,
        ratio_wh,
        perim / (h + w),
        r_mean / h,
        c_mean / w,
    ]  # 6

    # --- B. Intensités des régions faciales (8 features) ---
    regions = {
        'left_eye':  (30, 55,  20,  50),
        'right_eye': (30, 55,  78, 108),
        'nose':      (55, 85,  45,  83),
        'mouth':     (88, 118, 35,  95),
    }
    feats_regions = []
    for (y0, y1, x0, x1) in regions.values():
        m, s = _region_stats(face_img, y0, y1, x0, x1)
        feats_regions.extend([m, s])  # 2 × 4 = 8

    # --- C. Intensités globales (3 features) ---
    left_mean  = float(face_img[:, :w//2].mean()) / 255.0
    right_mean = float(face_img[:, w//2:].mean()) / 255.0
    global_mean = float(face_img.mean()) / 255.0
    feats_global = [left_mean, right_mean, global_mean]  # 3

    # --- D. Symétrie & gradient vertical/horizontal (4 features) ---
    left_half  = face_img[:, :w//2].astype(np.float32) / 255.0
    right_half = np.fliplr(face_img[:, w//2:].astype(np.float32) / 255.0)
    symmetry   = float(np.mean(np.abs(left_half - right_half)))  # 0 = symétrique

    top_half    = face_img[:h//2, :].astype(np.float32).mean() / 255.0
    bottom_half = face_img[h//2:, :].astype(np.float32).mean() / 255.0
    vert_grad   = float(top_half - bottom_half)

    # Rapport front / bas visage
    forehead = face_img[:30, :].astype(np.float32).mean() / 255.0
    chin     = face_img[100:, :].astype(np.float32).mean() / 255.0
    fb_ratio = float(forehead / chin) if chin > 0 else 1.0

    feats_sym = [symmetry, vert_grad, top_half, fb_ratio]  # 4

    # --- E. Histogramme global 6 bins (6 features) ---
    hist, _ = np.histogram(face_img, bins=6, range=(0, 256))
    hist_norm = (hist / hist.sum()).tolist()  # 6

    # --- F. Textures Sobel (3 features) ---
    sobelx = cv2.Sobel(face_img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(face_img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    grad_mean = float(grad_mag.mean()) / 1000.0
    grad_std  = float(grad_mag.std())  / 1000.0

    # Asymétrie du gradient gauche/droite
    grad_left  = float(grad_mag[:, :w//2].mean())
    grad_right = float(grad_mag[:, w//2:].mean())
    grad_asym  = abs(grad_left - grad_right) / (grad_left + grad_right + 1e-6)

    feats_texture = [grad_mean, grad_std, grad_asym]  # 3

    # --- Fusion ---
    feats = (feats_snake       # 6
           + feats_regions     # 8
           + feats_global      # 3
           + feats_sym         # 4
           + hist_norm         # 6
           + feats_texture)    # 3
                               # = 30

    assert len(feats) == FEAT_DIM, f"Vecteur {len(feats)}D au lieu de {FEAT_DIM}D"
    return np.array(feats, dtype=np.float32)