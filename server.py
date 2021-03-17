import cv2
import face_recognition
from flask import Flask, request, Response
import jsonpickle
import math
import numpy as np
import pickle


#
# Identificação facial
#

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

#
# Aplicação Flask
#
app = Flask(__name__)


"""
@app.route('/api/teste', methods=['POST'])
def teste():
    r = request

    # Converte string da imagem para uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decodifica imagem
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Faz algo....

    # Retorna mensagem para o cliente
    response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
                }
    # COndifica resposta com jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")
"""

# Rota para identificação facial
@app.route('/api/identificacao', methods=['POST'])
def identificacao():
    r = request
    suspeito = "Erro"
    # Converte string da imagem para uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decodifica imagem
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Obtêm minuncias

    try:
        face_minuncias = face_recognition.face_encodings(img)[0]
    except:
        pass
    else:
        aux_identificacao = procurados.identificaPessoaMD(face_minuncias)
        print(aux_identificacao)
        if  aux_identificacao["distancia"]<0.5:
            suspeito = aux_identificacao['nome']
        else:
            suspeito = "Não identificado"

    # Retorna mensagem para o cliente
    response = {'suspeito': suspeito}

    # Codifica resposta com jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


#
# Inicia aplicação Flask
#
app.run(host="0.0.0.0", port=5000, debug=False)
