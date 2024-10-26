from flask import Flask, request, render_template, redirect, url_for
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
import numpy as np
import cv2
import os
import gdown

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

# Ruta de carga del modelo desde Google Drive
model_url = 'https://drive.google.com/uc?id=1WEZ60x_yPY-gPv8ugoq_qDLTSAD541zc'  # Reemplaza con el ID correcto
model_path = 'modelo/model_VGG16_v4.keras'

# Descarga el modelo si no existe
if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

# Carga el modelo
model = load_model(model_path)

# Ruta de subida de imágenes
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ruta principal
@app.route("/", methods=["GET", "POST"])
def home():
    try:
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

            # Realiza la predicción
            preds = model.predict(img)
            predicted_class_index = np.argmax(preds)

            # Asegúrate de que el índice esté dentro del rango
            if 0 <= predicted_class_index < len(names):
                predicted_class_name = names[predicted_class_index]
                confidence_percentage = preds[0][predicted_class_index] * 100
            else:
                predicted_class_name = "Clase desconocida"
                confidence_percentage = 0.0

            # Renderiza el resultado
            return render_template("index.html", 
                                   prediction=predicted_class_name, 
                                   confidence=f"{confidence_percentage:.2f}")

        # Si es una solicitud GET, renderiza la interfaz inicial
        return render_template("index.html")

    except Exception as e:
        # Captura cualquier excepción y renderiza el error
        return render_template("index.html", 
                               prediction="Error en la aplicación: " + str(e), 
                               confidence="0.00")

# Corre la aplicación
if __name__ == "__main__":
    app.run(debug=True)
