# Modelo de Classificação de Peixes (PyTorch/ResNet)

Este projeto treina uma rede neural convolucional (ResNet18 pré-treinada) para classificar imagens de peixes usando PyTorch.

O script `Modelo_ia.py` está configurado para:
* Carregar um dataset de imagens das pastas `./data/train` e `./data/test`.
* Aplicar transformações de redimensionamento e normalização.
* Dividir os dados de treino em conjuntos de treino e validação (80/20).
* Configurar um modelo ResNet18 pré-treinado e adaptar a camada final para o número de classes do dataset.
* Executar um loop de treinamento por 10 épocas, validando o modelo ao final de cada época.
* Salvar logs de treinamento na pasta `runs/` para visualização no TensorBoard.

## 🛠️ Tecnologias Utilizadas

* **Python 3**
* **PyTorch**
* **Torchvision**
* **scikit-learn** (para `train_test_split`)
* **Matplotlib** (para visualização de amostras)
* **TensorBoard** (para logging)

## 📦 Instalação

1.  **Clone o repositório e entre na pasta:**
    ```bash
    git clone https://github.com/pedrogaleano15/IA.git
    cd IA/Modelo_ia
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Como Executar

### 1. Preparação dos Dados

O script **requer** que os dados de imagem estejam organizados no formato `ImageFolder` dentro da pasta `data/`, com uma subpasta por classe em `train/` e `test/`. O dataset já incluído no repositório (`data/train` e `data/test`) contém três classes de peixes ornamentais:

```
data/
├── train/
│   ├── acara-bandeira/
│   ├── carpa/
│   └── platy-laranja/
└── test/
    ├── acara-bandeira/
    ├── carpa/
    └── platy-laranja/
```

O arquivo `v4_train_test.zip` é a origem compactada desses mesmos dados.

### 2. Executar o treinamento

```bash
python Modelo_ia.py
```

O script imprime as contagens de treino/validação/teste, treina o modelo por 10 épocas e grava as métricas de perda e acurácia em `runs/experimento_ia` (visualizável com `tensorboard --logdir runs`).
