from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input

# Nombres de clases (solo mostrando algunos por espacio)
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

# Inicializar Flask y configurar CORS
app = Flask(__name__)
CORS(app)

# Configuración de rutas
UPLOAD_FOLDER = './uploaded_images'
app.config.from_mapping(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    ALLOWED_EXTENSIONS={'png', 'jpg', 'jpeg', 'gif'}
)

# Asegurar que el directorio de imágenes existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cargar el modelo
model_path = os.path.join(os.path.dirname(__file__), 'model_VGG16_v4.keras')
model = load_model(model_path)

def allowed_file(filename):
    """Verificar si el archivo tiene una extensión permitida."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/api/', methods=['GET'])
def get_example():
    """Ejemplo de endpoint GET."""
    return jsonify({"message": "Este es un ejemplo de respuesta GET"})

@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    """Endpoint para subir y clasificar una imagen."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            # Procesar imagen y predecir clase
            image = cv2.resize(cv2.imread(filepath), (224, 224))
            input_data = preprocess_input(np.expand_dims(image, axis=0))
            preds = model.predict(input_data)

            # Obtener predicción y confianza
            class_index = np.argmax(preds)
            class_name = names[class_index]
            confidence = preds[0][class_index] * 100

            return jsonify({
                "message": f'Clase predicha: {class_name}, Confianza: {confidence:.2f}%',
                "file_path": filepath
            }), 200
        except Exception as e:
            return jsonify({"error": f"Error al procesar la imagen: {str(e)}"}), 500
    else:
        return jsonify({"error": "Tipo de archivo no permitido"}), 400

@app.route('/api/post_example', methods=['POST'])
def post_example():
    """Ejemplo de endpoint POST."""
    data = request.get_json()
    return jsonify({
        "message": "Datos recibidos correctamente",
        "data": data
    })

@app.errorhandler(404)
def not_found(error):
    """Manejador de error 404."""
    return jsonify({"error": "Recurso no encontrado"}), 404

@app.route('/')
def serve_interface():
    """Servir archivo HTML desde la raíz."""
    return send_from_directory('.', 'index2.html')

if __name__ == '__main__':
    # Usar el puerto asignado por el entorno si está disponible
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
