from __future__ import print_function
import cv2
import face_recognition
from flask import Flask, jsonify, render_template, request, Response
import json
import math
import numpy as np
import pygame
import requests
import time


#
# Teste
#

"""
addr = 'http://localhost:5000'
test_url = addr + '/api/identificacao'

# Inicializa câmera
camera = cv2.VideoCapture(0)
anterior = time.time()

# Prepara cabeçalhos para requisição http
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('obama.jpg')

# Codifica imagem como jpeg
_, img_encoded = cv2.imencode('.jpg', img)

# Envia http request com imagem e recebe resposta
response = requests.post(test_url, data=img_encoded.tobytes(), headers=headers)

# Decodifica resposta
print(json.loads(response.text))
"""


# --------------------------------------------------------------------------

# Constantes
fpsLimite = 0.2 # número de frames capturados por segundo.
m = 0.75 # Fator de redução do frame.
addr = 'http://localhost:5000'
identificacao_url = addr + '/api/identificacao'


# Prepara cabeçalhos para requisição http
content_type = 'image/jpeg'
headers = {'content-type': content_type}

# Carrega o som
pygame.mixer.pre_init(44100, 16, 2, 4096) # frequência, tamanho, canais, buffer
pygame.init()
som = pygame.mixer.Sound('alerta.wav')


#Inicializa a aplicação Flask
app = Flask(__name__)

'''
Inicializa a captura pelo OpenCV
Câmera ip: rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp'
Webcam: cv2.VideoCapture(0)
http://help.angelcam.com/en/articles/372649-finding-rtsp-addresses-for-ip-cameras-nvrs-dvrs
'''

# print(retornaCameraIndices())

camera = cv2.VideoCapture(-1)
if not camera.isOpened():
    raise IOError("Não consegui abrir a webcam!")

# Define resolução
# camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
# camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

# Seta início do timer
anterior = time.time()


# Adiciona janela e gera os frames para a câmera
suspeito = ''
def gen_frames():
    global anterior, suspeito
    fonte = cv2.FONT_HERSHEY_DUPLEX

    # detectaFace = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    # if detectaFace.empty():
    #    raise IOError('Erro ao carregar haarcascade_frontalface_alt.xml')

    while True:
        agora = time.time()
        if (agora - anterior) > fpsLimite:
            c0 = time.time()
            sucesso, frame = camera.read()  # Lê frame
            if sucesso:
                anterior = agora
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pequeno_frame = cv2.resize(rgb_frame,
                                           (0, 0),
                                           fx=m, fy=m)
                gray_frame = cv2.cvtColor(pequeno_frame, cv2.COLOR_BGR2GRAY) # Converte para cinza

                # Detecção de face utilizando opencv
                # faces = detectaFace.detectMultiScale(pequeno_frame,
                #                                      scaleFactor=1.4,
                #                                      minNeighbors=1,
                #                                      minSize=(50, 50))

                # Detecta as coordenadas da caixa das faces detectadas.
                c1 = time.time()
                faces = face_recognition.face_locations(gray_frame)
                print('Detecção de faces: ', time.time() - c1)
                n_faces = len(faces)
                print('Número de faces: ', n_faces)

                if n_faces==1:

                    (topo, direita, base, esquerda) = faces[0]
                    imagem_face = pequeno_frame[topo:base, esquerda:direita]

                    # Codifica imagem como jpeg
                    _, img_encoded = cv2.imencode('.jpg', imagem_face)

                    # face_minuncias = face_recognition.face_encodings(pequeno_frame,
                    #                                                  faces)[0]

                    try:
                        # Envia http request com imagem e recebe resposta
                        c1 = time.time()
                        resposta = requests.post(identificacao_url, data=img_encoded.tostring(), headers=headers)
                        print('Identificação: ', time.time() - c1)

                        # Decodifica resposta
                        identificacao = json.loads(resposta.text)
                    except:
                        print('Erro no servidor da API.')
                    else:

                        c1 = time.time()
                        # aux_identificacao = procurados.identificaPessoaMD(face_minuncias)
                        # print("Busca de indivíduo: ",
                        #       time.time() - c1)

                        topo = int(topo//m)
                        direita = int(direita//m)
                        base = int(base//m)
                        esquerda = int(esquerda//m)

                        if  identificacao["suspeito"]=="Não identificado":
                            print("Não identificado!")
                            cv2.rectangle(frame,
                                          (esquerda, topo),
                                          (direita, base),
                                          (0, 255, 0),
                                          2)
                        elif identificacao["suspeito"]== "Erro":
                            pass
                        else:
                            cv2.rectangle(frame,
                                          (esquerda, topo),
                                          (direita, base),
                                          (0, 0, 255),
                                          2)
                            cv2.putText(frame,
                                        identificacao["suspeito"],
                                        (esquerda + 7, base - 7),
                                        fonte,
                                        1,
                                        (0, 0, 255),
                                        1,
                                        cv2.LINE_AA)
                            som.play()
                            time.sleep(3)

                elif n_faces==0:
                    print("Nenhuma face detectada!")
                else:
                    print("Mais de uma face detectada!")

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                saida = (b'--frame\r\n'
                         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                print(' --> Tempo de processamento: ', time.time() - c0)
                yield saida # Concatena e mostra resultado


# Define uma rota para a aplicação web
@app.route('/')
def index():
    return render_template('index.html')


# Define uma rota para o stream de video
@app.route('/streamVideo')
def streamVideo():
    frames = gen_frames()
    return Response(frames, mimetype='multipart/x-mixed-replace; boundary=frame')


# Carrrega pessoas.
"""
@app.route('/processaBase')
def processaBase():
    global procurados

    # Cria grupo de pessoas.
    procurados = grupoPessoas("procurados")
    for pasta in os.listdir(diretorio_trabalho +
                            '/individuos'):
        aux_pessoa = Pessoa(pasta)
        for imagem in os.listdir(diretorio_trabalho +
                                 '/individuos/' +
                                 pasta +
                                 ''):
            aux_pessoa.calculaMinuncia(diretorio_trabalho +
                                       '/individuos/' +
                                       pasta +
                                       '/' +
                                       imagem)
            print(pasta + '/' + imagem + ' processado')
        procurados.inserePessoa(aux_pessoa)
    print(procurados.mostraNomes())

    # Salva o objeto
    with open('procurados.pkl', 'wb') as f:
        pickle.dump(procurados, f)
    print('Arquivo procurados.pkl salvo!')
    return 'Base processada!'
"""

# Inicia o servidor Flask
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False)
