import os
import zipfile
import shutil
import numpy as np
import cv2

# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------
CARPETA_BASE = "dataset_flores"
ARCHIVO_ZIP = "flores_rojas_azules.zip"
NUM_IMAGENES_POR_CLASE = 50
TAMANIO_IMAGEN = (128, 128)  # ancho x alto

# Colores en BGR (OpenCV usa BGR)
COLOR_ROJO = (0, 0, 255)
COLOR_AZUL = (255, 0, 0)

# ----------------------------------------------------------------------
# Función para generar una imagen de "flor"
# ----------------------------------------------------------------------
def generar_flor(color_petalo, fondo=None):
    """
    Crea una imagen sintética de una flor simple.
    """
    if fondo is None:
        # Fondo aleatorio (verde, blanco, gris, cielo)
        fondos_posibles = [
            (0, 255, 0),      # verde
            (255, 255, 255),  # blanco
            (128, 128, 128),  # gris
            (255, 255, 200)   # beige
        ]
        fondo = fondos_posibles[np.random.randint(len(fondos_posibles))]

    img = np.full((TAMANIO_IMAGEN[1], TAMANIO_IMAGEN[0], 3), fondo, dtype=np.uint8)

    centro = (TAMANIO_IMAGEN[0] // 2 + np.random.randint(-10, 10),
              TAMANIO_IMAGEN[1] // 2 + np.random.randint(-10, 10))

    # Pétalos: círculos alrededor del centro
    for i in range(5 + np.random.randint(0, 3)):
        angulo = i * (2 * np.pi / 5) + np.random.uniform(-0.2, 0.2)
        radio_petalo = 20 + np.random.randint(-5, 10)
        distancia = 25 + np.random.randint(-5, 5)
        pt = (int(centro[0] + distancia * np.cos(angulo)),
              int(centro[1] + distancia * np.sin(angulo)))
        cv2.circle(img, pt, radio_petalo, color_petalo, -1)

    # Centro de la flor (amarillo o blanco)
    color_centro = (0, 255, 255) if np.random.random() > 0.5 else (255, 255, 255)
    cv2.circle(img, centro, 12 + np.random.randint(-3, 5), color_centro, -1)

    # Añadir algo de ruido para variabilidad
    ruido = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = cv2.add(img, ruido)

    # Pequeño desenfoque para suavizar
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img

# ----------------------------------------------------------------------
# Generación del dataset
# ----------------------------------------------------------------------
print("🌱 Generando dataset sintético de flores...")

if os.path.exists(CARPETA_BASE):
    shutil.rmtree(CARPETA_BASE)
os.makedirs(CARPETA_BASE, exist_ok=True)

clases = {
    'rojas': COLOR_ROJO,
    'azules': COLOR_AZUL
}

for clase, color in clases.items():
    carpeta_destino = os.path.join(CARPETA_BASE, clase)
    os.makedirs(carpeta_destino, exist_ok=True)
    print(f"\n📥 Creando imágenes para clase '{clase}'...")

    for i in range(NUM_IMAGENES_POR_CLASE):
        img = generar_flor(color)
        nombre_archivo = f"{clase}_{i:03d}.jpg"
        ruta_guardado = os.path.join(carpeta_destino, nombre_archivo)
        cv2.imwrite(ruta_guardado, img)

    print(f"   ✅ {clase}: {NUM_IMAGENES_POR_CLASE} imágenes generadas.")

# ----------------------------------------------------------------------
# Crear archivo ZIP
# ----------------------------------------------------------------------
print(f"\n🗜️ Creando archivo ZIP '{ARCHIVO_ZIP}'...")
with zipfile.ZipFile(ARCHIVO_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(CARPETA_BASE):
        for file in files:
            ruta_completa = os.path.join(root, file)
            ruta_relativa = os.path.relpath(ruta_completa, os.path.dirname(CARPETA_BASE))
            zipf.write(ruta_completa, ruta_relativa)

print("\n✅ ¡Proceso completado!")
print(f"Dataset sintético listo: '{ARCHIVO_ZIP}' (con carpetas 'rojas' y 'azules').")