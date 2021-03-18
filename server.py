import cv2
import face_recognition
from flask import Flask, request, Response
import jsonpickle
import numpy as np
from oorf import Pessoa, grupoPessoas
import pickle


#
# Identificação facial
#


padrao_reconhecimento = 0.5


# Carrega arquivo de exemplo
print("Carregando base de dados...")
with open('procurados.pkl', 'rb') as f:
     procurados = pickle.load(f)
print("Feito!")
print("Procurados: ", procurados.mostraNomes())
# print("Minuncias: ", procurados.mostraMinuncias())

#
# Aplicação Flask
#
app = Flask(__name__)


# Rota para identificação facial
@app.route('/api/identificacao', methods=['POST'])
def identificacao():
    r = request

    try:
        # Converte string da imagem para uint8
        nparr = np.fromstring(r.data, np.uint8)

        # Decodifica a imagem
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Converte para RGB
        # rgb_img = img[:, :, ::-1]

    except:
        print('Erro na imagem')
    else:
        cv2.imwrite('frame.jpg', img)

    try:
        # Obtêm minuncias
        face_minuncias = face_recognition.face_encodings(rgb_img)
    except:
        print('Erro na obtenção das minuncias')
        suspeito = "Erro "
    else:
        if len(face_minuncias)==1:
            aux_identificacao = procurados.identificaPessoaMD(face_minuncias[0])
            if  aux_identificacao["distancia"]<padrao_reconhecimento:
                suspeito = aux_identificacao['nome']
            else:
                suspeito = "Não identificado"
        else:
            suspeito = "Erro - número de faces diferente de 1"

    # Retorna mensagem para o cliente
    response = {'suspeito': suspeito}

    # Codifica resposta com jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


#
# Inicia aplicação Flask
#
app.run(host="0.0.0.0", port=5000, debug=False)
