import cv2
import os

# Carregar o classificador pré-treinado para detecção facial
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar a câmera
cap = cv2.VideoCapture(0)

# Pasta contendo imagens de referência
reference_folder = 'img/'

# Lista de caminhos para as imagens de referência na pasta
reference_images = [os.path.join(reference_folder, file) for file in os.listdir(reference_folder) if file.endswith(('.jpg', '.jpeg', '.png'))]

print(reference_images)

# Converter imagens de referência para escala de cinza
reference_grays = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY) for img in reference_images]


while True:
    # Capturar frame por frame
    ret, frame = cap.read()

    # Converter a imagem para escala de cinza para a detecção facial
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar faces no frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Desenhar retângulos ao redor das faces detectadas
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Comparar a face detectada com cada imagem de referência
        roi_gray = gray[y:y+h, x:x+w]
        for i, reference_gray in enumerate(reference_grays):
            if cv2.compareHist(cv2.calcHist([roi_gray], [0], None, [256], [0, 256]), cv2.calcHist([reference_gray], [0], None, [256], [0, 256]), cv2.HISTCMP_CORREL) > 0.8:
                cv2.putText(frame, f'Mesma pessoa {i+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                break
        else:
            # print('oi')
            cv2.putText(frame, 'Diferente pessoa', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Exibir o frame resultante
    cv2.imshow('Video', frame)

    # Encerrar o loop quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
