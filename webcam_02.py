import cv2
import face_recognition
from flask import Flask, jsonify, render_template, request, Response
import math
import numpy as np
import os
import pickle
import pygame 
import time
import webbrowser


fpsLimite = 1.2 # número de frames capturados por segundo.
m = 0.75 # Fator de redução do frame.


"""
Definir volume no terminal
amixer set PCM unmute
amixer set PCM 100%

Para evitar timeout da câmera
/#bin/bash
sudo rmmod uvcvideo
sudo modprobe uvcvideo nodrop=1 timeout=5000 quirks=0x80
"""

diretorio_trabalho = os.getcwd()

# Carrega o som
pygame.mixer.pre_init(44100, 16, 2, 4096) # frequência, tamanho, canais, buffer
pygame.init()
som = pygame.mixer.Sound('alerta.wav')

"""
import beepy
beep(sound=1)

"""

class Pessoa:
    def __init__(self, nome='Não definido'):
        self.nome = nome
        self.minuncias = []

    def calculaMinuncia(self, endereco_imagem):  
        imagem = face_recognition.load_image_file(endereco_imagem)
        minuncia = face_recognition.face_encodings(imagem)[0]
        self.minuncias.append(minuncia)     
    
    def minunciaMatch(self, face_minuncias):
        dist = []
        for aux_minuncias in self.minuncias:
            dist.append(np.linalg.norm(face_minuncias-aux_minuncias))
        # res = statistics.median(dist)
        res = min(dist)
        return res


class grupoPessoas:
    def __init__(self, nome_grupo='Não definido'):
        self.nome_grupo = nome_grupo
        self.pessoas = []
    
    def inserePessoa(self, pessoa):
        self.pessoas.append(pessoa)
        
    def buscaPessoa(self, nome):
        pessoa = None
        for aux_pessoa in self.pessoas:
            if aux_pessoa.nome==nome:
                pessoa = aux_pessoa
        return pessoa              
        
    def mostraMinuncias(self):
        minuncias = []
        nomes = []
        for aux_pessoa in self.pessoas:
            minuncias = minuncias + aux_pessoa.minuncias
            nomes = nomes + [aux_pessoa.nome]*len(aux_pessoa.minuncias)
        return {'minuncias': minuncias, 'nomes': nomes}
    
    def mostraNomes(self):
        nomes = []
        for aux_pessoa in self.pessoas:
            nomes = nomes + [aux_pessoa.nome]
        return nomes
    
    def calibraClassificador(self):
        self.clf = svm.SVC(gamma='scale')
        aux_minuncias = self.mostraMinuncias()
        self.clf.fit(aux_minuncias["minuncias"], aux_minuncias["nomes"])
        return self.clf.score(aux_minuncias["minuncias"], aux_minuncias["nomes"])
    
    def identificaPessoaSVM(self, face_minuncias):
        nome = self.clf.predict([face_minuncias])[0]
        escore = self.buscaPessoa(nome).minunciaMatch(face_minuncias)
        return {'nome': nome, 'escore':escore}
    
    def identificaPessoaMD(self, face_minuncias):
        nome = None
        distancia = math.inf
        for aux_pessoa in self.pessoas:
            aux_distancia = aux_pessoa.minunciaMatch(face_minuncias)
            if aux_distancia<distancia:
                distancia = aux_distancia
                nome = aux_pessoa.nome
        return {'nome': nome, 'distancia':distancia}


# Carrega arquivo de exemplo
print("Carregando base de dados...")
with open('procurados.pkl', 'rb') as f:
     procurados = pickle.load(f)
print("Feito!")
print("Procurados: ", procurados.mostraNomes())


#Inicializa a aplicação Flask
app = Flask(__name__)

'''
Inicializa a captura pelo OpenCV

Câmera ip: rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' 
Webcam: cv2.VideoCapture(0)
http://help.angelcam.com/en/articles/372649-finding-rtsp-addresses-for-ip-cameras-nvrs-dvrs
'''

camera = cv2.VideoCapture(0)
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
                    try:
                        c1 = time.time()
                        face_minuncias = face_recognition.face_encodings(pequeno_frame,
                                                                         faces)[0]
                        print("Cálculo de minuncias: ", time.time() - c1)
                    except:
                        pass
                    else:
                        c1 = time.time()
                        aux_identificacao = procurados.identificaPessoaMD(face_minuncias)
                        print("Busca de indivíduo: ",
                              time.time() - c1)
                        
                        (topo, direita, base, esquerda) = faces[0]
                        topo = int(topo//m)
                        direita = int(direita//m)
                        base = int(base//m)
                        esquerda = int(esquerda//m)
                        
                        
                        if  aux_identificacao["distancia"]<0.5:
                            suspeito = aux_identificacao['nome']
                            cv2.rectangle(frame,
                                          (esquerda, topo),
                                          (direita, base),
                                          (0, 0, 255),
                                          2)
                            cv2.putText(frame,
                                        aux_identificacao["nome"],
                                        (esquerda + 7, base - 7),
                                        fonte,
                                        1,
                                        (0, 0, 255),
                                        1,
                                        cv2.LINE_AA)
                            som.play()
                        else:
                            print("Não identificado!")
                            cv2.rectangle(frame,
                                          (esquerda, topo),
                                          (direita, base),
                                          (0, 255, 0),
                                          2)

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


# Inicia o servidor Flask
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False)