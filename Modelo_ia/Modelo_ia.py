# Nome do arquivo: modelo_ia.py
# ===================================================================
# 1. IMPORTAÇÃO DE BIBLIOTECAS
# ===================================================================
# importação de bibliotecas
import torch          # Biblioteca do pytorch principal
from torch import nn  # Redes Neurais (neural network)
from torch.utils.data import DataLoader # manipular as imagens banco de dados
from torchvision import datasets,models # Utilizar modelos
                                       # já existentes
# Transforms -> trabalha com imagens
import torchvision.transforms as transforms
import matplotlib.pyplot as plt # plotar/exibir as imagens
from torch.utils.data import Subset # conjuntos de dados

from sklearn.model_selection import train_test_split

# biblioteca para armzenar log
from torch.utils.tensorboard import SummaryWriter

# ===================================================================
# 2. HIPERPARÂMETROS
# ===================================================================
# hiperparâmetros utilizados no modelo de IA

taxa_aprendizagem = 0.01 # o quão relevante para alterar pesos
momento = 0.2 # armazena as informações dos pesos

tamanho_lote = 16 # quantidade de imagens (lote/batch)
perc_val = 0.2 # (20%) percentual de imagens utilizadas para validação
# arquitetura
nome_rede = 'resnet' # não necessita de tantos dados
                     # de entrada para classificar
tamanho_imagens = 224 # resolução das imagens (tamanho)
# pasta raiz do projeto
pasta_base = "./"
# pasta dos dados (banco de imagens)
# IMPORTANTE: Garanta que esta pasta exista ou descomente o download abaixo
pasta_data = pasta_base+"data/"
print("Vai ler das imagens de: ", pasta_data)
pasta_treino = pasta_data+"train"
pasta_teste = pasta_data+"test"

# ===================================================================
# (Opcional) DOWNLOAD E PREPARAÇÃO DO DATASET (se necessário)
# ===================================================================
# Descomente estas linhas se precisar baixar e extrair o dataset
# import os
# import zipfile
# import requests # Pode precisar instalar: pip install requests
#
# zip_url = "https://drive.google.com/uc?export=download&id=1aW5so-0XAvXlWzpKkvsergw6907Vb7JE"
# zip_path = os.path.join(pasta_data, "v4_train_test.zip")
#
# if not os.path.exists(pasta_data):
#     os.makedirs(pasta_data)
#
# if not os.path.exists(zip_path):
#     print(f"Baixando dataset de {zip_url}...")
#     response = requests.get(zip_url, stream=True)
#     with open(zip_path, "wb") as f:
#         for chunk in response.iter_content(chunk_size=8192):
#             f.write(chunk)
#     print("Download completo.")
# else:
#     print("Arquivo zip já existe.")
#
# if not os.path.exists(pasta_treino) or not os.path.exists(pasta_teste):
#     print(f"Extraindo {zip_path}...")
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(pasta_data)
#     print("Extração completa.")
# else:
#     print("Pastas train/test já existem.")


# ===================================================================
# 3. PREPARAÇÃO DO CONJUNTO DE DADOS
# ===================================================================
# Definir transformações nas imagens:
transform = transforms.Compose([transforms.Resize((tamanho_imagens, tamanho_imagens)),
                                                  transforms.ToTensor()
                              ])
# Preparar o banco de imagens
try:
    training_val_data = datasets.ImageFolder(root=pasta_treino,
                                             transform=transform)
    test_data = datasets.ImageFolder(root=pasta_teste,
                                     transform=transform)
except FileNotFoundError:
    print(f"ERRO: Pastas {pasta_treino} ou {pasta_teste} não encontradas.")
    print("Certifique-se de que a pasta 'data' existe e contém 'train' e 'test'.")
    print("Você pode precisar descomentar a seção de download no código.")
    exit()

# Separar as imagens de Treinamento e Validação
train_idx, val_idx = train_test_split(list(range(len(training_val_data))), test_size=perc_val)
training_data = Subset(training_val_data, train_idx)
val_data = Subset(training_val_data, val_idx)

# Criar os DataLoaders
train_dataloader = DataLoader(training_data, batch_size=tamanho_lote, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=tamanho_lote, shuffle=True)
# Criar DataLoader para teste (para avaliação final, não usado no treino/val)
test_dataloader = DataLoader(test_data, batch_size=tamanho_lote, shuffle=False)


# ===================================================================
# 4. INFORMAÇÕES DO CONJUNTO
# ===================================================================
print("\n--- Informações do Dataset ---")
# Mostrar informações do primeiro lote de validação
try:
    x_batch, y_batch = next(iter(val_dataloader))
    print(f"Tamanho do lote de imagens: {x_batch.shape[0]}")
    print(f"Dimensões das imagens (C, H, W): {x_batch.shape[1:]}")
    print(f"Tamanho do lote de classes: {y_batch.shape[0]}")
except StopIteration:
    print("AVISO: DataLoader de validação está vazio. Verifique o dataset.")

# Contagem total
total_imagens=len(training_data)+len(val_data)+len(test_data)
print(f"\nTotal de imagens: {total_imagens}")
print(f"Total de imagens de treinamento: {len(training_data)}  ({100*(len(training_data)/total_imagens):.2f}%)")
print(f"Total de imagens de validação: {len(val_data)}  ({100*(len(val_data)/total_imagens):.2f}%)")
print(f"Total de imagens de teste: {len(test_data)}  ({100*(len(test_data)/total_imagens):.2f}%)")

# Mapeamento de classes
labels_map = {v: k for k, v in test_data.class_to_idx.items()}
print('Classes: ', labels_map)

# ===================================================================
# 5. MOSTRAR ALGUMAS IMAGENS (Opcional)
# ===================================================================
mostrar_imagens = False # Mude para True se quiser ver as imagens ao rodar
if mostrar_imagens and len(training_data) > 0:
    print("\n--- Amostra de Imagens de Treino ---")
    figure = plt.figure(figsize=(8,8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        if i > len(training_data): break
        img_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[img_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.permute(1, 2, 0)) # Ajusta a ordem dos canais para Matplotlib
    plt.show()

# ===================================================================
# 6. DEFINIÇÃO DA REDE NEURAL
# ===================================================================
print("\n--- Configurando a Rede Neural ---")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

total_classes = len(labels_map)

if nome_rede == "resnet":
    # Usando weights=... ao invés do 'pretrained' depreciado
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Ajusta a camada final para o número de classes do dataset
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, total_classes)
else:
    print(f"ERRO: Rede '{nome_rede}' não suportada neste script.")
    exit()

model = model.to(device)
# print(model) # Descomente se quiser ver a estrutura completa do modelo

# ===================================================================
# 7. FUNÇÃO DE PERDA, OTIMIZADOR E LOGS
# ===================================================================
otimizador = torch.optim.SGD(model.parameters(), lr=taxa_aprendizagem, momentum=momento)
funcao_perda = nn.CrossEntropyLoss()

# Cria o objeto SummaryWriter para logs do TensorBoard
log_dir = 'runs/experimento_ia' # Nome da pasta para os logs
writer = SummaryWriter(log_dir)
print(f"Logs do TensorBoard serão salvos em: {log_dir}")


# ===================================================================
# 8. FUNÇÕES DE TREINO E VALIDAÇÃO
# ===================================================================
def train(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train() # Modo de treinamento
    train_loss, train_correct = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Computa a predição e a perda
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Acumula perda e acertos
        train_loss += loss.item()
        train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 4 == 0: # Imprime a cada 4 lotes
            loss_val, current = loss.item(), (batch + 1) * len(X)
            print(f"  Perda Treino: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches
    train_accuracy = train_correct / size
    # Log para TensorBoard
    writer.add_scalar('Perda/Treino', train_loss, epoch)
    writer.add_scalar('Acurácia/Treino', train_accuracy, epoch)
    return train_loss, train_accuracy

def validation(dataloader, model, loss_fn, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # Modo de avaliação
    val_loss, val_correct = 0, 0

    with torch.no_grad(): # Não calcula gradientes na validação
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    val_loss /= num_batches
    val_accuracy = val_correct / size
    print(f"\n  Resultado na Validação:")
    print(f"  Acurácia: {(100*val_accuracy):>0.1f}%, Perda Média: {val_loss:>8f}")
    # Log para TensorBoard
    writer.add_scalar('Perda/Validação', val_loss, epoch)
    writer.add_scalar('Acurácia/Validação', val_accuracy, epoch)
    return val_loss, val_accuracy


# ===================================================================
# 9. EXECUTANDO O TREINAMENTO
# ===================================================================
epocas = 10 # Número de épocas para treinar

print("\n--- Iniciando Treinamento ---")
for t in range(epocas):
    print(f"\nÉpoca {t+1}/{epocas}\n-------------------------------")
    train_loss, train_acc = train(train_dataloader, model, funcao_perda, otimizador, t)
    print(f"  Resultado no Treino Época {t+1}: Acurácia: {(100*train_acc):>0.1f}%, Perda: {train_loss:>8f}")
    validation(val_dataloader, model, funcao_perda, t)

print("\n--- Treinamento finalizado! ---")

# Fecha o writer do TensorBoard
writer.close()

# (Opcional) Salvar o modelo treinado
# torch.save(model.state_dict(), "modelo_ia_final.pth")
# print("Modelo salvo em modelo_ia_final.pth")