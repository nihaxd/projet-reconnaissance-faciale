import os

# Chemins
DATA_DIR    = "data"
CSV_PATH    = os.path.join(DATA_DIR, "dataset.csv")
TESTS_PATH  = os.path.join(DATA_DIR, "tests_results.csv")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")

# Hyperparamètres
THRESHOLD       = 3.5   # seuil distance euclidienne
FACE_SIZE       = 128    # taille image normalisée (pixels)
MIN_CAPTURES    = 20     # captures recommandées par personne

# Snake (active_contour)
SNAKE_ALPHA  = 0.01
SNAKE_BETA   = 10
SNAKE_GAMMA  = 0.001
SNAKE_ITER   = 250
SNAKE_POINTS = 200

# Descripteur
FEAT_DIM        = 30     # dimension finale du vecteur