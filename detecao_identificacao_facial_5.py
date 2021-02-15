# Identificação facial.

import os
import face_recognition
# import picamera
import numpy as np
import cv2
from sklearn import svm
import statistics
import math

# from PIL import Image, ImageDraw

diretorio_trabalho = '/home/pi/Documents/Identificação facial/Indivíduos/'
os.chdir(diretorio_trabalho)

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
        mediana = statistics.median(dist)      
        return mediana


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

cv2.startWindowThread()
cv2.namedWindow("preview")

# Cria pessoas e insere suas minuncias.
# Cria "Jaime".
print("Processando Jaime...")
jaime = Pessoa('Jaime')
jaime.calculaMinuncia(diretorio_trabalho+'Jaime/WhatsApp Image 2021-02-10 at 10.59.35.jpeg')
jaime.calculaMinuncia(diretorio_trabalho+'Jaime/WhatsApp Image 2021-02-10 at 10.59.36.jpeg')
jaime.calculaMinuncia(diretorio_trabalho+'Jaime/WhatsApp Image 2021-02-10 at 10.59.36 (1).jpeg')
jaime.calculaMinuncia(diretorio_trabalho+'Jaime/WhatsApp Image 2021-02-10 at 10.59.37.jpeg')
jaime.calculaMinuncia(diretorio_trabalho+'Jaime/WhatsApp Image 2021-02-10 at 10.59.37 (1).jpeg')

# Cria "Obama".
#print("Processando Obama...")
#obama = Pessoa('Obama')
#obama.calculaMinuncia(diretorio_trabalho+'Obama/obama.jpg')

# Cria "Priscila".
#print("Processando Priscila...")
#priscila = Pessoa('Priscila')
#priscila.calculaMinuncia(diretorio_trabalho+'Priscila/WhatsApp Image 2021-02-10 at 10.58.17.jpeg')
#priscila.calculaMinuncia(diretorio_trabalho+'Priscila/WhatsApp Image 2021-02-10 at 10.58.18.jpeg')
#priscila.calculaMinuncia(diretorio_trabalho+'Priscila/WhatsApp Image 2021-02-10 at 10.58.18 (1).jpeg')
#priscila.calculaMinuncia(diretorio_trabalho+'Priscila/WhatsApp Image 2021-02-10 at 10.58.19.jpeg')

# Cria grupo de pessoas.
grupo_teste = grupoPessoas("POLITEC")

# Insere pessoas no grupo.
grupo_teste.inserePessoa(jaime)
#grupo_teste.inserePessoa(priscila)
#grupo_teste.inserePessoa(obama)

# Inicializa a câmera do Raspberry Pi.
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FPS, 0.1)
print(video_capture.get(cv2.CAP_PROP_FPS))

while True:
    # Registra um frame.
    ret, frame = video_capture.read()
    
    # Redimensiona o frame para 3/4.
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
       
    # Converte de BGR para RGB.
    output = np.array(small_frame[:, :, ::-1])
    frame = np.array(frame[:, :, ::-1])
    
    # Detecta faces.
    faces_locacoes = face_recognition.face_locations(output)
    face_minuncias = None
    n_faces = len(faces_locacoes)
    if n_faces==1:
        print("Face detectada.")
        face_minuncias = face_recognition.face_encodings(output)[0]
        aux_identificacao = grupo_teste.identificaPessoaMD(face_minuncias)
        if  aux_identificacao["distancia"]<0.5:
            print(aux_identificacao)
            print(faces_locacoes[0])
            
            (topo, direita, base, esquerda) = faces_locacoes[0]
            topo = 4*topo
            direita = 4*direita
            base = 4*base
            esquerda = 4*esquerda
            
            # Desenha o retângulo da face.
            cv2.rectangle(frame,
                          (esquerda, topo),
                          (direita, base),
                          (0, 0, 255),
                          1)

            # Desenha o nome.
            cv2.rectangle(frame,
                          (esquerda, base - 35),
                          (direita, base),
                          (0, 0, 255),
                          cv2.FILLED)
            fonte = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame,
                        aux_identificacao["nome"],
                        (esquerda + 6, base - 6),
                        fonte,
                        1,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA)
            
            # Pressione 'f' para sair.
            if cv2.waitKey(1) & 0xFF == ord('q'):
               break
        else:
            print("Não identificado!")
    elif n_faces==0:
        print("Nenhuma face detectada!")
    else:
        print("Mais de uma face detectada!")
    
    # Mostra o frame.
    cv2.imshow('preview', frame)
        
# Libera a webcam
video_capture.release()
cv2.destroyAllWindows()        