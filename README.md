# Integrantes
- Juan Camilo Suarez
- Andres Felipe Saavedra
- Juan Esteban Santiago

# Clasificador de Imágenes con FastAPI

API genérica para entrenar y usar modelos de clasificación de imágenes (Regresión Logística, SVM, Árbol de Decisión) basada en histogramas de color.

## Endpoints principales

| Método | Ruta                  | Descripción                         |
|--------|-----------------------|-------------------------------------|
| POST   | `/train`              | Entrena un modelo desde un ZIP      |
| POST   | `/predict/{model}`    | Clasifica una imagen                |
| GET    | `/models`             | Lista modelos cargados              |
| DELETE | `/models/{model}`     | Elimina un modelo                   |

<img width="1920" height="1020" alt="Captura de pantalla 2026-04-18 120248" src="https://github.com/user-attachments/assets/78e1e526-53af-4c98-94b6-87d4b8cca842" />
<img width="1920" height="1020" alt="Captura de pantalla 2026-04-18 115810" src="https://github.com/user-attachments/assets/2d8ab64f-e2ec-4077-b2e9-a01ca079a461" />
