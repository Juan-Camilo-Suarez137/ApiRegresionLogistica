import os
import tempfile
import zipfile
import shutil
from typing import List, Dict, Optional

import numpy as np
import cv2
import joblib  # Para serialización de modelos
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------------------------
# Definiciones de la API (Pydantic)
# ----------------------------------------------------------------------
class PredictionResponse(BaseModel):
    model_name: str
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]

class TrainingResponse(BaseModel):
    model_name: str
    status: str
    train_accuracy: float
    test_accuracy: float
    message: str

# ----------------------------------------------------------------------
# Configuración de persistencia
# ----------------------------------------------------------------------
MODELS_DIR = os.getenv("MODELS_DIR", "./models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Extractores de características
# ----------------------------------------------------------------------
def default_histogram_extractor(image_path, channels=(1, 2), bins=(256, 256)):
    """Extrae histograma 2D de los canales especificados."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar: {image_path}")
    hist = cv2.calcHist([img], list(channels), None, list(bins), [0, 256, 0, 256])
    return hist.flatten()

# ----------------------------------------------------------------------
# Clase ImageClassifier
# ----------------------------------------------------------------------
class ImageClassifier:
    SUPPORTED_MODELS = {
        'logistic': LogisticRegression,
        'svm': SVC,
        'decision_tree': DecisionTreeClassifier
    }

    def __init__(self, model_name='logistic', class_names=None,
                 feature_extractor=default_histogram_extractor,
                 model_params=None):
        self.model_name = model_name.lower()
        self.class_names = class_names if class_names else ['clase_0', 'clase_1']
        self.feature_extractor = feature_extractor
        self.model_params = model_params if model_params else {}

        # Ajustes para SVM: activar probabilidades
        if self.model_name == 'svm' and 'probability' not in self.model_params:
            self.model_params['probability'] = True

        self.scaler = StandardScaler()
        self.model = None
        self._is_trained = False

    def _create_model(self):
        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Modelo '{self.model_name}' no soportado.")
        model_class = self.SUPPORTED_MODELS[self.model_name]
        return model_class(**self.model_params)

    def load_images_from_folders(self, base_dir, class_names=None):
        if class_names is None:
            class_names = self.class_names

        X, y = [], []
        for label in class_names:
            folder = os.path.join(base_dir, label)
            if not os.path.isdir(folder):
                print(f"Advertencia: Carpeta no encontrada: {folder}")
                continue
            for fname in os.listdir(folder):
                full_path = os.path.join(folder, fname)
                try:
                    features = self.feature_extractor(full_path)
                    X.append(features)
                    y.append(label)
                except Exception as e:
                    print(f"Error procesando {full_path}: {e}")
        return X, y

    def train(self, X, y, test_size=0.33, random_state=1):
        X = np.array(X)
        y = np.array(y)

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )

        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        self._is_trained = True

        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)
        return train_acc, test_acc

    def predict(self, image_path):
        if not self._is_trained:
            raise RuntimeError("Modelo no entrenado.")
        features = self.feature_extractor(image_path)
        features = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        pred = self.model.predict(features_scaled)
        return pred[0]

    def predict_proba(self, image_path):
        if not self._is_trained:
            raise RuntimeError("Modelo no entrenado.")
        if not hasattr(self.model, 'predict_proba'):
            raise RuntimeError("El modelo no soporta probabilidades.")
        features = self.feature_extractor(image_path)
        features = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        proba = self.model.predict_proba(features_scaled)[0]
        return {cls: prob for cls, prob in zip(self.model.classes_, proba)}

    def save(self, filepath):
        """Serializa el clasificador a disco."""
        data = {
            'model_name': self.model_name,
            'class_names': self.class_names,
            'model_params': self.model_params,
            'scaler': self.scaler,
            'model': self.model,
            '_is_trained': self._is_trained
        }
        joblib.dump(data, filepath)

    @classmethod
    def load(cls, filepath, feature_extractor=default_histogram_extractor):
        """Carga un clasificador desde disco."""
        data = joblib.load(filepath)
        instance = cls(
            model_name=data['model_name'],
            class_names=data['class_names'],
            feature_extractor=feature_extractor,
            model_params=data['model_params']
        )
        instance.scaler = data['scaler']
        instance.model = data['model']
        instance._is_trained = data['_is_trained']
        return instance

# ----------------------------------------------------------------------
# Almacenamiento global de modelos (en memoria)
# ----------------------------------------------------------------------
models_store: Dict[str, ImageClassifier] = {}

# ----------------------------------------------------------------------
# Cargar modelos existentes al arrancar
# ----------------------------------------------------------------------
for filename in os.listdir(MODELS_DIR):
    if filename.endswith(".joblib"):
        model_name = filename[:-7]  # Quita la extensión .joblib
        filepath = os.path.join(MODELS_DIR, filename)
        try:
            models_store[model_name] = ImageClassifier.load(filepath)
            print(f"Modelo '{model_name}' cargado desde {filepath}")
        except Exception as e:
            print(f"Error al cargar modelo {filepath}: {e}")

# ----------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------
app = FastAPI(title="Clasificador de Imágenes Genérico")

@app.post("/train", response_model=TrainingResponse)
async def train_model(
    model_name: str = Form(..., description="Nombre único para el modelo (ej. 'modelo_saludable')"),
    class_names: List[str] = Form(..., description="Lista de nombres de clases (ej. ['saludable','nosaludable'])"),
    model_type: str = Form('logistic', description="Tipo de modelo: 'logistic', 'svm', 'decision_tree'"),
    test_size: float = Form(0.33, description="Proporción de datos para prueba"),
    random_state: int = Form(1),
    zip_file: UploadFile = File(..., description="Archivo ZIP con carpetas por clase")
):
    """Entrena un nuevo modelo a partir de un ZIP con imágenes y lo guarda en disco."""
    if model_name in models_store:
        raise HTTPException(status_code=400, detail=f"El modelo '{model_name}' ya existe. Usa otro nombre.")

    if model_type not in ImageClassifier.SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Tipo de modelo no soportado. Opciones: {list(ImageClassifier.SUPPORTED_MODELS.keys())}")

    if len(class_names) < 2:
        raise HTTPException(status_code=400, detail="Se requieren al menos dos clases.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "dataset.zip")
        with open(zip_path, "wb") as f:
            content = await zip_file.read()
            f.write(content)

        extract_dir = os.path.join(tmp_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="El archivo no es un ZIP válido.")

        classifier = ImageClassifier(model_name=model_type, class_names=class_names)

        try:
            X, y = classifier.load_images_from_folders(extract_dir, class_names)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al cargar imágenes: {str(e)}")

        if len(X) == 0:
            raise HTTPException(status_code=400, detail="No se encontraron imágenes válidas en el dataset.")

        try:
            train_acc, test_acc = classifier.train(X, y, test_size=test_size, random_state=random_state)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error durante el entrenamiento: {str(e)}")

        # Guardar modelo en memoria y en disco
        models_store[model_name] = classifier
        filepath = os.path.join(MODELS_DIR, f"{model_name}.joblib")
        classifier.save(filepath)

    return TrainingResponse(
        model_name=model_name,
        status="entrenado",
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        message=f"Modelo '{model_name}' entrenado exitosamente y guardado en disco."
    )


@app.post("/predict/{model_name}", response_model=PredictionResponse)
async def predict_image(
    model_name: str,
    image: UploadFile = File(..., description="Imagen a clasificar")
):
    """Predice la clase de una imagen utilizando un modelo previamente entrenado."""
    if model_name not in models_store:
        raise HTTPException(status_code=404, detail=f"Modelo '{model_name}' no encontrado.")

    classifier = models_store[model_name]

    # Guardar imagen temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
        content = await image.read()
        tmp_img.write(content)
        tmp_img_path = tmp_img.name

    try:
        predicted_class = classifier.predict(tmp_img_path)
        proba_dict = classifier.predict_proba(tmp_img_path)
        confidence = proba_dict.get(predicted_class, 0.0)
    except Exception as e:
        os.unlink(tmp_img_path)
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")
    finally:
        os.unlink(tmp_img_path)

    return PredictionResponse(
        model_name=model_name,
        predicted_class=predicted_class,
        confidence=confidence,
        all_probabilities=proba_dict
    )


@app.get("/models", response_model=List[str])
async def list_models():
    """Lista los nombres de los modelos actualmente cargados en memoria."""
    return list(models_store.keys())


@app.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Elimina un modelo de la memoria y del disco."""
    if model_name in models_store:
        del models_store[model_name]
        # Eliminar archivo
        filepath = os.path.join(MODELS_DIR, f"{model_name}.joblib")
        if os.path.exists(filepath):
            os.remove(filepath)
        return {"status": "eliminado", "model_name": model_name}
    else:
        raise HTTPException(status_code=404, detail=f"Modelo '{model_name}' no encontrado.")