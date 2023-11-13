









#  fim
import cv2

# Carregar o classificador pré-treinado para detecção facial
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar a câmera
cap = cv2.VideoCapture(0)

reference = cv2.imread('img/tttt.jpg')
reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

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

        roi_gray = gray[y:y+h, x:x+w]
        for 
        if cv2.compareHist(cv2.calcHist([roi_gray], [0], None, [256], [0, 256]), cv2.calcHist([reference_gray], [0], None, [256], [0, 256]), cv2.HISTCMP_CORREL) > 0.8:
            cv2.putText(frame, 'Mesma pessoa', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Diferente pessoa', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Exibir o frame resultante
    cv2.imshow('Video', frame)

    # Encerrar o loop quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
