from flask import Flask, request, render_template
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
import numpy as np
import cv2
import os
import gdown  # Biblioteca para descargar archivos de Google Drive

# Inicializa la app Flask
app = Flask(__name__)

# Lista de nombres de aves
names = ['Amazona Alinaranja', 'Amazona de San Vicente', 'Amazona Mercenaria', 'Amazona Real',
         'Aratinga de Pinceles', 'Aratinga de Wagler', 'Aratinga Ojiblanca', 'Aratinga Orejigualda',
         'Aratinga Pertinaz', 'Batará Barrado', 'Batará Crestibarrado', 'Batara Crestinegro',
         'Batará Mayor', 'Batará Pizarroso Occidental', 'Batará Unicolor', 'Cacatua Ninfa', 
         'Catita Frentirrufa', 'Cotorra Colinegra', 'Cotorra Pechiparda', 'Cotorrita Alipinta',
         'Cotorrita de Anteojos', 'Guacamaya Roja', 'Guacamaya Verde', 'Guacamayo Aliverde',
         'Guacamayo azuliamarillo', 'Guacamayo Severo', 'Hormiguerito Coicorita Norteño',
         'Hormiguerito Coicorita Sureño', 'Hormiguerito Flanquialbo', 'Hormiguerito Leonado',
         'Hormiguerito Plomizo', 'Hormiguero Azabache', 'Hormiguero Cantor', 'Hormiguero de Parker',
         'Hormiguero Dorsicastaño', 'Hormiguero Guardarribera Oriental', 'Hormiguero Inmaculado',
         'Hormiguero Sencillo', 'Hormiguero Ventriblanco', 'Lorito Amazonico', 'Lorito Cabecigualdo',
         'Lorito de fuertes', 'Loro Alibronceado', 'Loro Cabeciazul', 'Loro Cachetes Amarillos',
         'Loro Corona Azul', 'Loro Tumultuoso', 'Ojodefuego Occidental', 'Periquito Alas Amarillas',
         'Periquito Australiano', 'Periquito Barrado', 'Tiluchí Colilargo', 'Tiluchí de Santander',
         'Tiluchi Lomirrufo']

# Configuración de la carpeta de subida de imágenes
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Enlace al modelo en Google Drive (ID del archivo)
MODEL_ID = "1WEZ60x_yPY-gPv8ugoq_qDLTSAD541zc"
MODEL_PATH = "modelo_vgg16.keras"

# Función para descargar el modelo desde Google Drive si no está disponible localmente
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Descargando el modelo...")
        url = f"https://drive.google.com/uc?id={MODEL_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# Descargar el modelo
download_model()
model = load_model(MODEL_PATH)

# Ruta principal
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Obtiene la imagen subida
        image = request.files["image"]
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
        image.save(image_path)

        # Procesa la imagen
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))  # Ajusta al tamaño esperado por VGG16
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        try:
            # Realiza la predicción
            preds = model.predict(img)
            predicted_class_index = np.argmax(preds)

            # Verifica si el índice está dentro del rango de nombres
            if 0 <= predicted_class_index < len(names):
                predicted_class_name = names[predicted_class_index]
                confidence_percentage = preds[0][predicted_class_index] * 100
            else:
                predicted_class_name = "Clase desconocida"
                confidence_percentage = 0.0
        except Exception as e:
            return render_template("index.html", 
                                   prediction="Error en la predicción", 
                                   confidence="0.00")

        # Renderiza el resultado
        return render_template("index.html", 
                               prediction=predicted_class_name, 
                               confidence=f"{confidence_percentage:.2f}")

    # Si es una solicitud GET, renderiza la interfaz inicial
    return render_template("index.html")

# Corre la aplicación
if __name__ == "__main__":
    app.run(debug=True)
