import cv2
import numpy as np
from tensorflow.keras.models import load_model
import face_recognition

# Cargar el modelo previamente entrenado para la detección de emociones
modelo_emociones = load_model("modeloCNN.h5")
modelomlp = load_model("modeloMLP.h5")

# Etiquetas de las imágenes del dataset
labels = ['bored', 'engaged', 'excited', 'focused', 'interested', 'relaxed']

# Función para detectar caras y predecir emociones en tiempo real
def detectar_y_predecir_emocionesCNN(frame, modelo):
    # Convertir el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar caras en el frame
    caras = face_recognition.face_locations(frame)

    if caras is not None:
    
     # Para cada cara detectada
        for (x, y, w, h) in caras:
        # Extraer la región de interés (ROI) correspondiente a la cara
            for location in caras:
                top, right, bottom, left = location
                roi = frame[top:bottom, left:right]
            

            if roi.size!=0:
        
        # Redimensionar la ROI a 150x150 (para que coincida con el input_shape del modelo)
                roi_resized = cv2.resize(roi, (150, 150))
        
        # Normalizar la imagen (dividiendo por 255)
                roi_normalized = roi_resized / 255.0
        
        # Realizar la predicción de emociones en la ROI
                roi_predicted = modelo.predict(np.expand_dims(roi_normalized, axis=0))#[0]
        
        # Obtener el índice de la etiqueta con mayor probabilidad
                idx_etiqueta = np.argmax(roi_predicted)
        
        # Obtener la etiqueta correspondiente
                etiqueta = labels[idx_etiqueta]
        
        # Mostrar la etiqueta en la pantalla
                #cv2.putText(frame, etiqueta, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, etiqueta, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
                #cv2.imshow('Emociones en tiempo real', roi)
        # Dibujar un rectángulo alrededor de la cara detectada
                #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

# Iniciar la captura de video desde la cámara web
video_capture = cv2.VideoCapture(0)

# Cargar el clasificador de Haar para la detección de caras
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capturar un frame de la cámara
    ret, frame = video_capture.read()

    # Detectar y predecir emociones en el frame
    detectar_y_predecir_emociones(frame, modelo_emociones)

    # Mostrar el frame resultante
    cv2.imshow('Emociones en tiempo real', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar las ventanas
video_capture.release()
cv2.destroyAllWindows()
