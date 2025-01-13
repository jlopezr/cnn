import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Obtener path del archivo python
import os
path = os.path.abspath("inference.py")
os.chdir(os.path.dirname(path))

# Cargar el modelo
model = tf.keras.models.load_model("model_cnn.h5")

# Leer el argumento de la línea de comandos
import sys

if len(sys.argv) != 2:
    imagen = "number.png"
else:
    imagen = sys.argv[1]

# Cargar y preprocesar la imagen
img = image.load_img(imagen, target_size=(28, 28), color_mode='grayscale')
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalizar la imagen

# Realizar la predicción
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

for i in range(10):
    print()

print(f"Analizando la imagen {imagen}")
print(f"La imagen contiene un número {predicted_class[0]}")