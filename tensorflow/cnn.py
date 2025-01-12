import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Cargar el dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar las imágenes (de 0-255 a 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Redimensionar las imágenes para CNN (agregar un canal)
x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn = x_test.reshape(-1, 28, 28, 1)

# Convertir las etiquetas a formato categórico (one-hot encoding)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# --------- CUDA ---------

# Listar los dispositivos disponibles
devices = tf.config.list_physical_devices()
print("Dispositivos disponibles:")
for device in devices:
    print(device)

# Verificar si hay GPUs disponibles
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("TensorFlow está utilizando la GPU.")
else:
    print("TensorFlow no está utilizando la GPU.")

# --------- Red Neuronal Clasica ---------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import time
import os

if os.path.exists('model_nn.h5'):
    model_nn = tf.keras.models.load_model('model_nn.h5')
else:
    # Definir la arquitectura
    model_nn = Sequential([
        Flatten(input_shape=(28, 28)),  # Aplana la imagen
        Dense(128, activation='relu'),  # Capa oculta
        Dense(64, activation='relu'),   # Otra capa oculta
        Dense(10, activation='softmax') # Capa de salida
    ])

    # Compilar el modelo
    model_nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    start_time = time.time()
    history_nn = model_nn.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Tiempo total de entrenamiento: {total_time:.2f} segundos")

    # Guardar el modelo
    model_nn.save('model_nn.h5')

# --------- Red Neuronal Convolucional (CNN) ---------

from tensorflow.keras.layers import Conv2D, MaxPooling2D
import time
import os

if os.path.exists('model_cnn.h5'):
    model_cnn = tf.keras.models.load_model('model_cnn.h5')
else:
    # Definir la arquitectura
    model_cnn = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Capa convolucional 1
        MaxPooling2D((2, 2)),                                            # Pooling 1
        Conv2D(64, (3, 3), activation='relu'),                          # Capa convolucional 2
        MaxPooling2D((2, 2)),                                            # Pooling 2
        Flatten(),                                                      # Aplana
        Dense(128, activation='relu'),                                  # Capa oculta
        Dense(10, activation='softmax')                                 # Capa de salida
    ])

    # Compilar el modelo
    model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    start_time = time.time()
    history_cnn = model_cnn.fit(x_train_cnn, y_train, validation_data=(x_test_cnn, y_test), epochs=10, batch_size=32)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Tiempo total de entrenamiento: {total_time:.2f} segundos")

    # Guardar el modelo
    model_cnn.save('model_cnn.h5')
    
# --------- Comparar resultados ---------

import matplotlib.pyplot as plt

# Gráfica de precisión
plt.plot(history_nn.history['accuracy'], label='NN - Entrenamiento')
plt.plot(history_nn.history['val_accuracy'], label='NN - Validación')
plt.plot(history_cnn.history['accuracy'], label='CNN - Entrenamiento')
plt.plot(history_cnn.history['val_accuracy'], label='CNN - Validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()

# Gráfica de pérdida
plt.plot(history_nn.history['loss'], label='NN - Entrenamiento')
plt.plot(history_nn.history['val_loss'], label='NN - Validación')
plt.plot(history_cnn.history['loss'], label='CNN - Entrenamiento')
plt.plot(history_cnn.history['val_loss'], label='CNN - Validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# --------- Comparar resultados ---------

from tensorflow.keras.utils import plot_model
from PIL import Image

# Dibujar la estructura del modelo de la red neuronal clásica con tamaño ajustado
plot_model(model_nn, to_file='model_nn.png', show_shapes=True, show_layer_names=True, dpi=96)
Image.open('model_nn.png').show()

# Dibujar la estructura del modelo de la red neuronal convolucional con tamaño ajustado
plot_model(model_cnn, to_file='model_cnn.png', show_shapes=True, show_layer_names=True, dpi=96)
Image.open('model_cnn.png').show()