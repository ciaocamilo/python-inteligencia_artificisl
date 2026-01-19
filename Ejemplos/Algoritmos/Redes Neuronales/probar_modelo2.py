# Ubicarse en la carpeta "Machine Learning\Ejemplos\Algoritmos\Redes Neuronales"
import tensorflow as tf
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# =========================
# 🔹 1. Cargar el modelo
# =========================
modelo = tf.keras.models.load_model("modelo_clasificador_perros_gatos.keras")
print("✅ Modelo cargado correctamente")

# =========================
# 🔹 2. Función para preparar imágenes
# =========================
def preparar_imagen(ruta, tamano=100):
    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta}")

    img = cv2.resize(img, (tamano, tamano))
    img = img / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img

# =========================
# 🔹 3. Carpetas de validación
# =========================
carpeta_validacion = r"cats_and_dogs_filtered\validation"
clases = ["cats", "dogs"]

resultados = []
y_true = []
y_pred = []

# =========================
# 🔹 4. Evaluar imágenes
# =========================
for clase in clases:
    carpeta = os.path.join(carpeta_validacion, clase)
    for archivo in os.listdir(carpeta):
        ruta = os.path.join(carpeta, archivo)

        try:
            img = preparar_imagen(ruta)
            prob = modelo.predict(img, verbose=0)[0][0]

            prediccion_clase = "dogs" if prob > 0.5 else "cats"
            es_correcto = (prediccion_clase == clase)

            resultados.append({
                "imagen": archivo,
                "clase_real": clase,
                "prediccion": prediccion_clase,
                "probabilidad_dog": float(prob),
                "correcto": es_correcto
            })

            y_true.append(0 if clase == "cats" else 1)
            y_pred.append(0 if prediccion_clase == "cats" else 1)

        except Exception as e:
            print(f"⚠️ Error con {ruta}: {e}")

# =========================
# 🔹 5. Guardar resultados a CSV
# =========================
df = pd.DataFrame(resultados)
df.to_csv("reporte_validacion.csv", index=False, encoding="utf-8")
print("📂 Reporte guardado en: reporte_validacion.csv")

# =========================
# 🔹 6. Matriz de confusión con porcentajes
# =========================
if y_true and y_pred:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_normalizada = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100  # porcentajes por fila

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_normalizada, interpolation="nearest", cmap=plt.cm.Blues)

    # Etiquetas
    ax.set_xticks(np.arange(len(clases)))
    ax.set_yticks(np.arange(len(clases)))
    ax.set_xticklabels(clases)
    ax.set_yticklabels(clases)
    ax.set_ylabel("Clase Real")
    ax.set_xlabel("Clase Predicha")
    plt.title("Matriz de Confusión (%)")

    # Mostrar valores en celdas
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                f"{cm[i, j]} ({cm_normalizada[i, j]:.1f}%)",
                ha="center", va="center",
                color="white" if cm_normalizada[i, j] > 50 else "black"
            )

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.show()

    precision = np.mean([t == p for t, p in zip(y_true, y_pred)]) * 100
    print(f"\n📊 Precisión en validación: {precision:.2f}% ({sum(np.array(y_true)==np.array(y_pred))}/{len(y_true)})")
else:
    print("⚠️ No se encontraron imágenes válidas para generar la matriz de confusión.")
