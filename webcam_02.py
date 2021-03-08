import cv2
import face_recognition
from flask import Flask, render_template, Response
import math
import numpy as np
import os
import pickle
import time


diretorio_trabalho = '/home/pi/Documents/Identificação facial/'
os.chdir(diretorio_trabalho)

detectaFace = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
if detectaFace.empty():
    raise IOError('Erro ao carregar haarcascade_frontalface_alt.xml')

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
    def __init__(self, nome='Não definido'):
        self.nome = nome
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
            minuncias = minuncias+aux_pessoa.minuncias
            nomes = nomes+[aux_pessoa.nome]*len(aux_pessoa.minuncias)
        return {'minuncias': minuncias, 'nomes': nomes}
    
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
fpsLimite = 1

# Adiciona janela e gera os frames para a câmera

def gen_frames():
    global anterior
    while True:
        sucesso, frame = camera.read()  # Lê frame
        agora = time.time()
        if not sucesso:
            break
        else:
            if (agora - anterior) > fpsLimite:
                anterior = agora
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)               
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Converte para cinza 
                pequeno_frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3) # Reduz para 30%
                
                c1 = time.time()
                faces = detectaFace.detectMultiScale(gray_frame,
                                                     scaleFactor=1.2,
                                                     minNeighbors=5,
                                                     minSize=(120, 120))
                print('Detecção de faces: ', time.time() - c1)
                n_faces = len(faces)
                print('Número de faces: ', n_faces)
                
                                
                if n_faces==1:
                    for (x,y,w,h) in faces:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) 
                        print('Altura: ', h, ', largura: ', w)
                            
                    c1 = time.time()
                    face_minuncias = face_recognition.face_encodings(pequeno_frame)[0]
                    print("Cálculo de minuncias: ", time.time() - c1)
                    c1 = time.time()
                    aux_identificacao = grupo_teste.identificaPessoaMD(face_minuncias)
                    print("Busca de indivíduo: ", time.time() - c1)
                    if  aux_identificacao["distancia"]<0.5:
                        print(aux_identificacao)
                    else:
                        print("Não identificado!")
                elif n_faces==0:
                    print("Nenhuma face detectada!")
                else:
                    print("Mais de uma face detectada!")
                    
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Concatena e mostra resultado
                    


# Carrega arquivo de exemplo
print("Carregando exemplo...")
with open('objs.pkl', 'rb') as f:
     grupo_teste = pickle.load(f)
print("Feito!")
 
 
 # Define uma rota para a aplicação web
@app.route('/')
def index():
    return render_template('index.html')

# Define uma rota para o stream de video
@app.route('/stream_video')
def stream_video():
    frames = gen_frames()
    return Response(frames, mimetype='multipart/x-mixed-replace; boundary=frame')

# Inicia o servidor Flask
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False)