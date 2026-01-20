import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =========================
# 🔹 Configuración general
# =========================
IMG_SIZE = 224

CLASES = {
    0: "❓ Ninguna de las anteriores",
    1: "🏜️ Desierto",
    2: "⛰️ Montaña",
    3: "🏖️ Playa"
}

# =========================
# 🔹 Cargar modelo (ruta robusta)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELO_PATH = os.path.join(
    BASE_DIR,
    "modelo_clasificador_playa_desierto_montana.keras"
)

modelo = tf.keras.models.load_model(MODELO_PATH)

# =========================
# 🔹 Preparar imagen
# =========================
def preparar_imagen(ruta):
    img = Image.open(ruta).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# 🔹 Clasificar imagen (salida 4 clases)
# =========================
def clasificar_imagen(ruta):
    img = preparar_imagen(ruta)
    salida = modelo.predict(img, verbose=0)
    salida = np.array(salida)

    if salida.ndim == 2 and salida.shape[1] == 4:
        probs = salida[0]
        idx = int(np.argmax(probs))
        confianza = float(probs[idx])
        etiqueta = CLASES.get(idx, "❓ Ninguna de las anteriores")
        return etiqueta, confianza, probs

    else:
        raise ValueError(f"Formato de salida no soportado: {salida.shape}")

# =========================
# 🔹 Mostrar gráfica
# =========================
def mostrar_grafica(probs):
    ax.clear()

    nombres = list(CLASES.values())
    valores = probs * 100

    ax.bar(nombres, valores)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Probabilidad (%)")
    ax.set_title("Confianza del modelo")

    for i, v in enumerate(valores):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)

    fig.tight_layout()
    canvas.draw()

# =========================
# 🔹 Cargar imagen desde UI
# =========================
def cargar_imagen():
    ruta = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Imágenes", "*.jpg *.jpeg *.png")]
    )

    if not ruta:
        return

    try:
        # Mostrar imagen
        img = Image.open(ruta)
        img.thumbnail((420, 280))
        img_tk = ImageTk.PhotoImage(img)
        label_imagen.config(image=img_tk)
        label_imagen.image = img_tk

        # Clasificar
        etiqueta, confianza, probs = clasificar_imagen(ruta)

        texto = f"🧠 Predicción final:\n{etiqueta}\n\n📊 Probabilidades:\n"
        for i, p in enumerate(probs):
            texto += f"• {CLASES[i]}: {p:.2%}\n"

        label_resultado.config(text=texto)
        mostrar_grafica(probs)

    except Exception as e:
        messagebox.showerror("Error", str(e))

# =========================
# 🔹 Interfaz gráfica
# =========================
root = tk.Tk()
root.title("Clasificador de Paisajes con IA")
root.geometry("650x900")
root.resizable(False, False)

# ---------- Frame superior ----------
frame_top = tk.Frame(root)
frame_top.pack(pady=15)

tk.Label(
    frame_top,
    text="🌍 Clasificador de Paisajes",
    font=("Arial", 18, "bold")
).pack(pady=5)

tk.Button(
    frame_top,
    text="📂 Cargar imagen",
    font=("Arial", 12),
    command=cargar_imagen
).pack(pady=10)

# ---------- Frame imagen ----------
frame_imagen = tk.Frame(root)
frame_imagen.pack(pady=15)

label_imagen = tk.Label(frame_imagen)
label_imagen.pack()

# ---------- Frame resultados ----------
frame_resultado = tk.Frame(root)
frame_resultado.pack(pady=15)

label_resultado = tk.Label(
    frame_resultado,
    font=("Arial", 12),
    justify="left",
    anchor="w"
)
label_resultado.pack()

# ---------- Frame gráfica ----------
frame_grafica = tk.Frame(root)
frame_grafica.pack(pady=20)

fig, ax = plt.subplots(figsize=(6, 3.5))
canvas = FigureCanvasTkAgg(fig, master=frame_grafica)
canvas.get_tk_widget().pack()

root.mainloop()
