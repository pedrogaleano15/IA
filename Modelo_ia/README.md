# Modelo de Classificação de Peixes (PyTorch/ResNet)

Este projeto treina uma rede neural convolucional (ResNet18 pré-treinada) para classificar imagens de peixes usando PyTorch.

O script `main.py` está configurado para:
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
    git clone [https://github.com/pedrogaleano15/GUSTAVO-IA.git](https://github.com/pedrogaleano15/GUSTAVO-IA.git)
    cd GUSTAVO-IA/modelopeixe
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

O script **requer** que os dados de imagem estejam organizados no formato `ImageFolder` dentro da pasta `data/`. (O arquivo `data.zip` enviado estava vazio, então você deve preenchê-lo localmente).

A estrutura de pastas esperada é: