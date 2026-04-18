import os
import zipfile
import shutil
from simple_image_download import Downloader

# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------
CARPETA_BASE = "dataset_flores"
ARCHIVO_ZIP = "flores_rojas_azules.zip"
NUM_IMAGENES_POR_CLASE = 50

CLASES = {
    'rojas': 'red flowers',
    'azules': 'blue flowers'
}

# ----------------------------------------------------------------------
# Descarga de imágenes reales desde Google
# ----------------------------------------------------------------------
print("🌱 Iniciando descarga de imágenes reales desde Google...")

# Limpiar carpeta base si existe
if os.path.exists(CARPETA_BASE):
    shutil.rmtree(CARPETA_BASE)
os.makedirs(CARPETA_BASE, exist_ok=True)

# Crear instancia del descargador (la que usa magic internamente)
downloader = Downloader()

for clase, keyword in CLASES.items():
    carpeta_destino = os.path.join(CARPETA_BASE, clase)
    os.makedirs(carpeta_destino, exist_ok=True)

    print(f"\n📥 Buscando y descargando '{clase}' (palabra clave: '{keyword}')...")

    try:
        # 1. Buscar las URLs de las imágenes (sin descargar aún)
        urls_encontradas = downloader.search_urls(
            keyword,
            limit=NUM_IMAGENES_POR_CLASE,
            verbose=False,
            cache=True
        )

        if not urls_encontradas:
            print(f"   ⚠️ No se encontraron imágenes para '{keyword}'.")
            continue

        # 2. Descargar las imágenes usando el caché obtenido
        downloader.download(
            keywords=keyword,
            limit=NUM_IMAGENES_POR_CLASE,
            verbose=False,
            download_cache=True   # Usa las URLs ya cacheadas
        )

        # 3. Mover las imágenes desde 'simple_images/{keyword}/' a nuestra estructura
        carpeta_origen = os.path.join("simple_images", keyword.replace(" ", "_"))
        if os.path.exists(carpeta_origen):
            archivos = os.listdir(carpeta_origen)
            for archivo in archivos:
                ruta_origen = os.path.join(carpeta_origen, archivo)
                ruta_destino = os.path.join(carpeta_destino, archivo)
                shutil.move(ruta_origen, ruta_destino)
            print(f"   ✅ {clase}: descargadas {len(archivos)} imágenes.")
            # Eliminar carpeta temporal vacía
            shutil.rmtree(carpeta_origen)
        else:
            print(f"   ⚠️ Carpeta temporal no encontrada para '{keyword}'.")

    except Exception as e:
        print(f"   ❌ Error con '{clase}': {e}")

# Limpiar carpeta raíz 'simple_images' si quedó vacía
if os.path.exists("simple_images"):
    shutil.rmtree("simple_images")

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
print(f"Dataset listo: '{ARCHIVO_ZIP}' (con carpetas 'rojas' y 'azules').")