# importacao de bibliotecas
import torch #biblioteca pyporch principal
from torch import nn #biblioteca de redes neurais
# É mais comum importar DataLoader diretamente, mas sua abordagem funciona
import torch.utils.data as DataLoader
from torchvision import datasets, models #utilizar modelos
import torchvision.transforms as transforms #utilizar transformacoes
import matplotlib.pyplot as plt #utilizar graficos
from torch.utils.data import Subset #utilizar subset
from sklearn.model_selection import train_test_split #utilizar train test split

# hiperparametros
# epocas -> quantidade de vezes que o modelo ira ver o dataset completo
# taxa de aprendizado -> o quanto o modelo ira ajustar os pesos a cada iteracao
# tolerancia -> o quanto o modelo ira tolerar erros
# paciencia -> o quanto o modelo ira tolerar erros antes de parar o treinamento
tamanho_lote = 16 #quantidade de imagens que o modelo ira ver antes de atualizar os pesos
perc_val = 0.2 #percentual do dataset que sera utilizado para validacao

nome_rede = 'resnet' #nome da rede neural que sera utilizada
tamanho_imagem = 224 #tamanho da imagem que sera utilizada (224 para resnet, 299 para inception)

# integracao com o Google drive
# from google.colab import drive
# drive.mount('/content/drive')

# baixar o dataset
# !curl -L -o v4_train_test.zip "https://drive.google.com/uc?export=download&id=1aW5so-0XAvXlWzpKkvsergw6907Vb7JE"
# !mkdir ./data/
# !mv v4*.zip ./data/
# %cd ./data/
# !unzip v4*.zip
# %cd ..

pasta_base = "./"
pasta_data = pasta_base + "data/"

print(f"vai ler as imagens de: {pasta_data}")
pasta_treino = pasta_data + "train"
pasta_teste = pasta_data + "test"

# definir tranformacoes nas imagens
# mudar o tamanho das imagens, transformar, normalizar
# redimencionalizar as imagens e transformar tensores

transformacoes = transforms.Compose([
    transforms.Resize((tamanho_imagem, tamanho_imagem)),
    transforms.ToTensor()
])
#preaparar os conjuntos de treinamento ,validação e teste
train_dataset = datasets.ImageFolder(pasta_treino, transform=transformacoes)
#teste
test_dataset = datasets.ImageFolder(pasta_teste, transform=transformacoes)

#preparar imagens de validacao
# CORREÇÃO APLICADA AQUI: Usando 'train_dataset' ao invés da variável inexistente 'training_val_data'
train_indx, val_idx = train_test_split(list(range(len(train_dataset))), test_size=perc_val)
print('train_idx:', train_indx)
print('val_idx:', val_idx) # Corrigido para 'val_idx' para uma impressão mais clara


#pegar as imagens com base no idx
training_data = Subset(train_dataset, train_indx)
val_data = Subset(train_dataset, val_idx)

# O código continua a partir daqui...
# Ex: Criar os DataLoaders
# train_loader = DataLoader.DataLoader(training_data, batch_size=tamanho_lote, shuffle=True)
# val_loader = DataLoader.DataLoader(val_data, batch_size=tamanho_lote)
# test_loader = DataLoader.DataLoader(test_dataset, batch_size=tamanho_lote)

print(f"\nDados carregados com sucesso!")
print(f"Total de imagens de treino/validação: {len(train_dataset)}")
print(f"Separando em {len(training_data)} para treino e {len(val_data)} para validação.")
print(f"Total de imagens de teste: {len(test_dataset)}")