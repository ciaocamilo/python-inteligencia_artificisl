# Ubicarse en la carpeta "Machine Learning\Ejemplos\Algoritmos\Redes Neuronales"
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# =========================
# 🔹 1. Cargar el modelo
# =========================
modelo = tf.keras.models.load_model("modelo_clasificador_perros_gatos.keras")
print("✅ Modelo cargado correctamente")

# =========================
# 🔹 2. Función para preparar imágenes
# =========================
def preparar_imagen(ruta, tamano=100):
    # Leer imagen en escala de grises
    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta}")

    # Redimensionar
    img = cv2.resize(img, (tamano, tamano))

    # Normalizar
    img = img / 255.0

    # Expandir dimensiones -> (1, 100, 100, 1)
    img = np.expand_dims(img, axis=(0, -1))

    return img

# =========================
# 🔹 3. Lista de imágenes de prueba
# =========================
imagenes_prueba = [
    r"cats_and_dogs_filtered\validation\cats\cat.2012.jpg",
    r"cats_and_dogs_filtered\validation\cats\cat.2494.jpg",
    r"cats_and_dogs_filtered\validation\dogs\dog.2499.jpg"
]

# =========================
# 🔹 4. Probar el modelo y mostrar resultados
# =========================
for ruta in imagenes_prueba:
    try:
        img = preparar_imagen(ruta)
        prediccion = modelo.predict(img, verbose=0)[0][0]

        etiqueta = "🐶 PERRO" if prediccion > 0.5 else "🐱 GATO"

        # Mostrar resultado en consola
        print(f"{ruta} → {etiqueta} ({prediccion:.2f})")

        # Mostrar imagen con título
        original = cv2.imread(ruta)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        plt.imshow(original)
        plt.title(f"{etiqueta} ({prediccion:.2f})")
        plt.axis("off")
        plt.show()

    except Exception as e:
        print(f"⚠️ Error con {ruta}: {e}")
