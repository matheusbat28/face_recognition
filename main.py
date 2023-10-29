import cv2
import mediapipe as mp

# Inicialize a webcam
webcam = cv2.VideoCapture(0)

# Inicialize o detector de rosto do MediaPipe
face_detector = mp.solutions.face_detection.FaceDetection()

# Inicialize o módulo de desenho do MediaPipe
drawing = mp.solutions.drawing_utils

# Loop principal para capturar e processar os quadros da webcam
while webcam.isOpened():
    # Captura um quadro da webcam
    ret, frame = webcam.read()

    # Verifica se a captura de quadro foi bem-sucedida
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Processa o quadro em busca de rostos
    results = face_detector.process(frame)

    # Se forem encontrados rostos no quadro
    if results.detections:
        for id, detection in enumerate(results.detections):
            # Desenha os retângulos ao redor dos rostos detectados no quadro
            drawing.draw_detection(frame, detection)

    # Exibe o quadro com as detecções
    cv2.imshow('Reconhecimento facial', frame)

    # Aguarda uma tecla ser pressionada por 1 milissegundo e verifica se é 'q' para sair do loop
    if cv2.waitKey(1) == ord('q'):
        break

    # Aguarda uma tecla ser pressionada por 1 milissegundo e verifica se é 's' para salvar uma captura de tela
    if cv2.waitKey(1) == ord('s'):
        cv2.imwrite('screenshot.jpg', frame)

# Libera a webcam e fecha a janela da exibição
webcam.release()
cv2.destroyAllWindows()
