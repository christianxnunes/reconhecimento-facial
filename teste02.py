import cv2
import face_recognition

# Pasta contendo imagens de referência
reference_folder = 'caminho/para/sua/pasta_com_imagens/'

# Lista de caminhos para as imagens de referência na pasta
image_paths = [os.path.join(reference_folder, file) for file in os.listdir(reference_folder) if file.endswith(('.jpg', '.jpeg', '.png'))]

# Lista para armazenar nomes das pessoas e os correspondentes encodings
known_face_names = []
known_face_encodings = []

# Iterar sobre imagens de referência e extrair encodings faciais
for image_path in image_paths:
    img = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(img)[0]  # Assumindo que há apenas uma face na imagem de referência
    known_face_encodings.append(encoding)
    known_face_names.append(os.path.splitext(os.path.basename(image_path))[0])

# Inicializar a câmera
cap = cv2.VideoCapture(0)

while True:
    # Capturar frame por frame
    ret, frame = cap.read()

    # Converter a imagem para RGB (necessário para o face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Encontrar todas as faces no frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Comparar a face detectada com as faces de referência
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Desconhecido"

        # Se encontrou correspondência, use o nome correspondente
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Desenhar retângulo e nome da pessoa no frame
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Exibir o frame resultante
    cv2.imshow('Video', frame)

    # Encerrar o loop quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()