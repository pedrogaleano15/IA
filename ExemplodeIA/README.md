# Modelo de Classificação de Imagens com PyTorch (ResNet)

Este é um script Python que serve como um modelo (template) para configurar um projeto de classificação de imagens. Ele usa a biblioteca `PyTorch` e carrega um modelo `ResNet` pré-treinado.

O script está configurado para:
* Definir hiperparâmetros (tamanho do lote, nome da rede).
* Definir transformações de imagem (redimensionar, transformar em tensor).
* Carregar os datasets de treino (`/train`) e teste (`/test`).
* Dividir o conjunto de treino em dados de treino e validação usando `scikit-learn`.

## 🛠️ Tecnologias Utilizadas

* **Python 3**
* **PyTorch**
* **Torchvision**
* **scikit-learn** (para `train_test_split`)
* **Matplotlib** (para visualização, embora não usado no script base)

## 📦 Instalação

1.  **Clone o repositório e entre na pasta:**
    ```bash
    git clone https://github.com/pedrogaleano15/IA.git
    cd IA/ExemplodeIA
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

Este script é um ponto de partida. Ele apenas **carrega e prepara os dados**. Ele não inicia o treinamento.

1.  **Prepare o Dataset:**
    O script espera que exista uma pasta `./data/` com subpastas `train/` e `test/`, seguindo o formato `ImageFolder` do PyTorch. As seções de download e descompactação de dados estão comentadas no código.

2.  **Execute o Script:**
    ```bash
    python ExemploModeloDeIA.py
    ```
    *(O script irá carregar os dados, dividi-los e imprimir as contagens de treino, validação e teste).*