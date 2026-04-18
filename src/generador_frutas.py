import os
import zipfile
import shutil
import numpy as np
import cv2

# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------
CARPETA_BASE = "dataset_frutas"
ARCHIVO_ZIP = "bananos_manzanas.zip"
NUM_IMAGENES_POR_CLASE = 50
TAMANIO_IMAGEN = (128, 128)  # ancho x alto

# ----------------------------------------------------------------------
# Función para generar un banano sintético
# ----------------------------------------------------------------------
def generar_banano(fondo=None):
    if fondo is None:
        fondos_posibles = [
            (200, 230, 200),  # verde claro
            (240, 240, 240),  # blanco
            (220, 220, 220),  # gris claro
            (180, 200, 220)   # azulado
        ]
        fondo = fondos_posibles[np.random.randint(len(fondos_posibles))]

    img = np.full((TAMANIO_IMAGEN[1], TAMANIO_IMAGEN[0], 3), fondo, dtype=np.uint8)

    # Color amarillo con variaciones
    color_banano = (0, np.random.randint(200, 255), np.random.randint(220, 255))

    centro = (TAMANIO_IMAGEN[0] // 2 + np.random.randint(-8, 8),
              TAMANIO_IMAGEN[1] // 2 + np.random.randint(-8, 8))
    ejes = (np.random.randint(30, 40), np.random.randint(15, 20))
    angulo = np.random.randint(-30, 30)

    cv2.ellipse(img, centro, ejes, angulo, 0, 360, color_banano, -1)

    # Curvatura adicional (forma de C)
    desplazamiento = 8
    centro2 = (centro[0] + int(desplazamiento * np.cos(np.radians(angulo))),
               centro[1] + int(desplazamiento * np.sin(np.radians(angulo))))
    cv2.ellipse(img, centro2, ejes, angulo, 0, 360, color_banano, -1)

    # Extremos oscuros
    cv2.circle(img, (centro[0] - ejes[0]//2, centro[1]), 3, (50, 50, 50), -1)
    cv2.circle(img, (centro[0] + ejes[0]//2, centro[1]), 3, (50, 50, 50), -1)

    # Ruido y desenfoque
    ruido = np.random.randint(0, 20, img.shape, dtype=np.uint8)
    img = cv2.add(img, ruido)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


# ----------------------------------------------------------------------
# Función para generar una manzana sintética
# ----------------------------------------------------------------------
def generar_manzana(fondo=None):
    if fondo is None:
        fondos_posibles = [
            (200, 230, 200),
            (240, 240, 240),
            (220, 220, 220),
            (180, 200, 220)
        ]
        fondo = fondos_posibles[np.random.randint(len(fondos_posibles))]

    img = np.full((TAMANIO_IMAGEN[1], TAMANIO_IMAGEN[0], 3), fondo, dtype=np.uint8)

    # Color rojo con variaciones
    color_manzana = (np.random.randint(0, 50), np.random.randint(0, 60), np.random.randint(200, 255))

    centro = (TAMANIO_IMAGEN[0] // 2 + np.random.randint(-6, 6),
              TAMANIO_IMAGEN[1] // 2 + np.random.randint(-6, 6))

    radio_x = np.random.randint(28, 36)
    radio_y = np.random.randint(30, 40)
    cv2.ellipse(img, centro, (radio_x, radio_y), 0, 0, 360, color_manzana, -1)

    # Tallo marrón
    tallo_inicio = (centro[0] - 2, centro[1] - radio_y + 5)
    tallo_fin = (centro[0] - 5 + np.random.randint(-3, 3), centro[1] - radio_y - 8)
    cv2.line(img, tallo_inicio, tallo_fin, (30, 60, 90), 3)

    # Hoja verde
    hoja_centro = (tallo_fin[0] + 6, tallo_fin[1] - 2)
    cv2.ellipse(img, hoja_centro, (8, 4), -30, 0, 360, (0, 150, 0), -1)

    # Brillo semitransparente
    overlay = img.copy()
    brillo_centro = (centro[0] - 8, centro[1] - 8)
    cv2.ellipse(overlay, brillo_centro, (8, 6), 0, 0, 360, (255, 255, 255), -1)
    img = cv2.addWeighted(img, 1, overlay, 0.25, 0)

    # Ruido y desenfoque
    ruido = np.random.randint(0, 20, img.shape, dtype=np.uint8)
    img = cv2.add(img, ruido)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


# ----------------------------------------------------------------------
# Generar dataset
# ----------------------------------------------------------------------
print("🍌🍎 Generando dataset sintético de frutas...")

if os.path.exists(CARPETA_BASE):
    shutil.rmtree(CARPETA_BASE)
os.makedirs(CARPETA_BASE, exist_ok=True)

# Bananos
carpeta_bananos = os.path.join(CARPETA_BASE, "bananos")
os.makedirs(carpeta_bananos, exist_ok=True)
print(f"\n📥 Creando {NUM_IMAGENES_POR_CLASE} imágenes de bananos...")
for i in range(NUM_IMAGENES_POR_CLASE):
    img = generar_banano()
    cv2.imwrite(os.path.join(carpeta_bananos, f"banano_{i:03d}.jpg"), img)
print(f"   ✅ Bananos: {NUM_IMAGENES_POR_CLASE} imágenes generadas.")

# Manzanas
carpeta_manzanas = os.path.join(CARPETA_BASE, "manzanas")
os.makedirs(carpeta_manzanas, exist_ok=True)
print(f"\n📥 Creando {NUM_IMAGENES_POR_CLASE} imágenes de manzanas...")
for i in range(NUM_IMAGENES_POR_CLASE):
    img = generar_manzana()
    cv2.imwrite(os.path.join(carpeta_manzanas, f"manzana_{i:03d}.jpg"), img)
print(f"   ✅ Manzanas: {NUM_IMAGENES_POR_CLASE} imágenes generadas.")

# ----------------------------------------------------------------------
# Crear ZIP
# ----------------------------------------------------------------------
print(f"\n🗜️ Creando archivo ZIP '{ARCHIVO_ZIP}'...")
with zipfile.ZipFile(ARCHIVO_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(CARPETA_BASE):
        for file in files:
            full = os.path.join(root, file)
            rel = os.path.relpath(full, os.path.dirname(CARPETA_BASE))
            zipf.write(full, rel)

print("\n✅ ¡Proceso completado!")
print(f"Dataset listo: '{ARCHIVO_ZIP}' (carpetas 'bananos' y 'manzanas').")