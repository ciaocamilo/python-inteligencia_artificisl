# cd \Ejemplos\Algoritmos\Redes Neuronales\Transferencia de aprendizaje
# conda activate tf-gpu
import tensorflow as tf
import numpy as np
import random
import os
from PIL import Image
import matplotlib.pyplot as plt

import matplotlib.font_manager as fm
from matplotlib import rcParams

emoji_font = fm.FontProperties(fname=r"C:\Windows\Fonts\seguiemj.ttf")
rcParams["font.family"] = emoji_font.get_name()

# =========================
# 🔹 1. Cargar el modelo
# =========================
modelo = tf.keras.models.load_model(
    "modelo_clasificador_bichofue_oso_anteojos.keras"
)
print("✅ Modelo cargado correctamente")

# =========================
# 🔹 2. Parámetros
# =========================
IMG_SIZE = 224
CLASES = {0: "🐦 BICHOFUÉ", 1: "🐻 OSO DE ANTEOJOS"}

BASE_DATASET = "dataset"
CARPETAS = {
    0: os.path.join(BASE_DATASET, "bichofue"),
    1: os.path.join(BASE_DATASET, "oso")
}

# =========================
# 🔹 3. Preparar imagen
# =========================
def preparar_imagen(ruta):
    img = Image.open(ruta).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # (1, 224, 224, 3)
    return img

# =========================
# 🔹 4. Función de predicción
# =========================
def categorizar(ruta):
    img = preparar_imagen(ruta)

    pred = modelo.predict(img, verbose=0)[0]   # ← vector de 2
    clase = np.argmax(pred)                    # ← índice correcto

    etiqueta = CLASES[clase]
    confianza = pred[clase]

    return etiqueta, confianza

# =========================
# 🔹 5. Seleccionar imágenes aleatorias
# =========================
imagenes_prueba = []

for clase, carpeta in CARPETAS.items():
    archivos = [
        os.path.join(carpeta, f)
        for f in os.listdir(carpeta)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    imagenes_prueba.extend(random.sample(archivos, k=5))

random.shuffle(imagenes_prueba)

# =========================
# 🔹 6. Evaluar y mostrar resultados
# =========================
for ruta in imagenes_prueba:
    try:
        etiqueta, confianza = categorizar(ruta)

        print(f"{ruta} → {etiqueta} ({confianza:.2%})")

        img = Image.open(ruta)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{etiqueta} ({confianza:.1%})")
        plt.show()

    except Exception as e:
        print(f"⚠️ Error con {ruta}: {e}")

# =========================
# 🔹 7. Evaluar imágenes SIN ETIQUETA (carpeta test)
# =========================
TEST_DATASET = "test"

print("\n🧪 Evaluando imágenes SIN etiqueta (carpeta test)\n")

imagenes_test = [
    os.path.join(TEST_DATASET, f)
    for f in os.listdir(TEST_DATASET)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

if len(imagenes_test) == 0:
    print("⚠️ No se encontraron imágenes en la carpeta test")
else:
    random.shuffle(imagenes_test)

    for ruta in imagenes_test:
        try:
            etiqueta, confianza = categorizar(ruta)

            print(f"[TEST] {os.path.basename(ruta)} → {etiqueta} ({confianza:.2%})")

            img = Image.open(ruta)
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"[TEST] {etiqueta} ({confianza:.1%})")
            plt.show()

        except Exception as e:
            print(f"⚠️ Error con {ruta}: {e}")
