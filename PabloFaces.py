import cv2
import numpy as np
from matplotlib import pyplot as plt

#Função para redimensionar uma imagem
def redim(img, largura): 
    alt = int(img.shape[0]/img.shape[1]*largura)
    img = cv2.resize(img, (largura, alt), interpolation =
    cv2.INTER_AREA)
    return img

#Cria o detector de faces baseado no XML
df = cv2.CascadeClassifier("modelo/haarcascade_frontalface_default.xml")

#Abre o vídeo
video_lido = cv2.VideoCapture("Videos/risadinha.mp4") # se passar 0 é a webcam


# Contador que serve de controle para nome das imagens salvas
contador = 0
# Loop main :D
while True:
    #read() retorna 1-Se houve sucesso e 2-O próprio frame
    (sucesso, frame) = video_lido.read() # Função com dois retornos
    if not sucesso: #final do vídeo
        break

    #reduz tamanho do frame para acelerar processamento
    frame = redim(frame, 320)
    #converte para tons de cinza
    frame_pb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detecta as faces no frame
    faces = df.detectMultiScale(frame_pb, scaleFactor = 1.1, minNeighbors=3, minSize=(5,5), flags=cv2.CASCADE_SCALE_IMAGE)
    frame_temp = frame.copy()
    
    #Histograma Frame
    histg = cv2.calcHist([frame_temp],[0],None,[256],[0,256])
    plt.plot(histg)
    plt.savefig('Hist/histograma' + str(contador) + '.png')
    plt.clf()
    # Cria os retângulos
    for (x, y, w, h) in faces:
        img = cv2.rectangle(frame_temp, (x, y), (x + w, y + h), (0, 255, 255), 2) # RGB
        # Salva cada frame com rosto
        roi_color = img[y:y+h, x:x+w]
    
        cv2.imwrite("Pessoas/image"+str(contador)+".png", roi_color) 
        
        histFace = cv2.calcHist([roi_color],[0],None,[256],[0,256])
        plt.plot(histFace)
        plt.savefig('HistFace/histograma' + str(contador) + '.png')
        plt.clf()
        # FIM HISTOGRAMA
        
    contador += 1
    #Exibe um frame redimensionado (com perca de qualidade)
    cv2.imshow("Encontrando faces...", redim(frame_temp, 640))

    #Espera que a tecla 's' seja pressionada para sair
    if cv2.waitKey(1) & 0xFF == ord("s"):
        break


#fecha streaming
video_lido.release()
cv2.destroyAllWindows()
