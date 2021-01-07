import cv2
import numpy as np

# cv2.CascadeClassifier([filename]) 
detecta_face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
detecta_olhos = cv2.CascadeClassifier('haarcascade_eye.xml')
detecta_nariz = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

if detecta_face.empty():
    raise IOError('Erro ao carregar haarcascade_frontalface_alt.xml')

if detecta_olhos.empty():
    raise IOError('Erro ao carregar haarcascade_eye.xml')
    
if detecta_nariz.empty():
    raise IOError('Erro ao carregar haarcascade_mcs_nose.xml')
    
# Inicializa objeto de captura de video
capture = cv2.VideoCapture(0)

while True:
    # Inicia a capturar frames
    ret, captura = capture.read()
        
    # cv2.resize(capturing, output image size, x scale, y scale, interpolation)
    frame_reduzido = cv2.resize(captura, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)  
    cinza = cv2.cvtColor(frame_reduzido, cv2.COLOR_BGR2GRAY)

    # cv2.CascadeClassifier.detectMultiScale(gray, scaleFactor, minNeighbors)
   
    # scaleFactor: Especifica o tamanho da imagem a ser reduzida.
        
    # minNeighbors: Especifica o numero de vizinhos em cada retangulo.
    # Valores mais altos resultam em número de deteccoes menor, mas com maior qualidade.
    
    # Face
    detecta_face = detecta_face.detectMultiScale(gray, 1.3, 5)

    # cv2.rectangle(image, (x1,y1), (x2,y2), color, thickness)
    for (x,y,w,h) in detecta_face:
        cv2.rectangle(frame_reduzido, (x,y), (x+w,y+h), (0,0,255), 10)    
        # Localiza a região de interesse (ROI)
        cinza_roi = cinza[y:y+h, x:x+w]
        cor_roi = resize_frame[y:y+h, x:x+w]
       
    # Olhos
    deteccao_olhos = detecta_olhos.detectMultiScale(cinza_roi)
        
    for (eye_x, eye_y, eye_w, eye_h) in detecta_olhos:
        cv2.rectangle(cor_roi,(eye_x,eye_y),(eye_x + eye_w, eye_y + eye_h),(255,0,0),5)
                       
    # Nariz
    deteccao_nariz = noise_detect.detectMultiScale(cinza_roi, 1.3, 5)

    for (nose_x, nose_y, nose_w, nose_h) in detecta_nariz:
        cv2.rectangle(color_roi, (nose_x, nose_y), (nose_x + nose_w, nose_y + nose_h), (0,255,0), 5)

    cv2.imshow("Deteccao em tempo real", resize_frame)

    c = cv2.waitKey(1)
    if c == 27: # ESC
       break

capture.release()

cv2.destroyAllWindows()
