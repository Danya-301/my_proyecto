from flask import Flask, request, render_template
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
import numpy as np
import cv2
import os

# Inicializa la app Flask
app = Flask(__name__)

# Lista de nombres de aves
names = [
    'Amazona Alinaranja', 'Amazona de San Vicente', 'Amazona Mercenaria', 'Amazona Real',
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
    'Tiluchi Lomirrufo'
]

# Carga el modelo
model_path = os.path.join(os.path.dirname(__file__), 'model_VGG16_v4.keras')
model = load_model(model_path)

# Ruta de subida de imágenes
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ruta principal
@app.route("/", methods=["GET", "POST"])
def home():
    try:
        if request.method == "POST":
            # Verifica si se subió un archivo
            if 'image' not in request.files:
                return render_template("index.html", prediction="No se subió ninguna imagen.", confidence="0.00")

            image = request.files["image"]
            if image.filename == '':
                return render_template("index.html", prediction="El archivo está vacío.", confidence="0.00")

            # Guarda la imagen
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
            image.save(image_path)

            # Procesa la imagen
            img = cv2.imread(image_path)
            img = cv2.resize(img, (224, 224))
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            # Realiza la predicción
            preds = model.predict(img)
            predicted_class_index = np.argmax(preds)

            # Verifica que el índice esté dentro del rango
            if 0 <= predicted_class_index < len(names):
                predicted_class_name = names[predicted_class_index]
                confidence_percentage = preds[0][predicted_class_index] * 100
            else:
                predicted_class_name = "Clase desconocida"
                confidence_percentage = 0.0

            return render_template("index.html", 
                                   prediction=predicted_class_name, 
                                   confidence=f"{confidence_percentage:.2f}")

        return render_template("index.html")

    except Exception as e:
        return render_template("index.html", 
                               prediction=f"Error en la aplicación: {str(e)}", 
                               confidence="0.00")

# Corre la aplicación
if __name__ == "__main__":
    app.run(debug=True)