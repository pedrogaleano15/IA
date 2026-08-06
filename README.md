# Inteligência Artificial — Estudos e Projetos Práticos

Repositório com implementações práticas de conceitos fundamentais de Inteligência Artificial, desenvolvidos durante a graduação em Engenharia da Computação na UCDB.

Cada subpasta é um projeto independente, com seu próprio `requirements.txt` e instruções de execução — não há um único ambiente compartilhado entre eles.

---

## Conteúdo

### BFS — Busca em Largura
**Pasta:** [`BFS/`](BFS/)

Resolvedor do clássico **8-Puzzle** implementado do zero, comparando três algoritmos de busca cega: **BFS** (Busca em Largura), **DFS** (Busca em Profundidade) e **IDS** (Aprofundamento Iterativo). Verifica a solubilidade do estado inicial (paridade de inversões) e exibe métricas de desempenho (nós expandidos, tamanho máximo da fronteira, tempo de execução) para cada algoritmo.

**Tecnologias:** Python 3, apenas biblioteca padrão (`collections`, `time`).

**Como executar:**
```bash
cd BFS
python BFS.py
```

---

### Rede Bayesiana (CLI + relatórios em PDF)
**Pasta:** [`rede_bayesiana/`](rede_bayesiana/)

Rede Bayesiana para inferência probabilística sobre fatores de risco de câncer de pulmão (poluição, tabagismo, enfisema, sintomas, etc.). Menu interativo em terminal permite rodar um cenário de exemplo, cadastrar novas evidências (salvas em `answers.json`) e gerar relatórios em PDF com os resultados e gráficos de cada consulta.

**Conceitos aplicados:** probabilidade condicional, teorema de Bayes, inferência por eliminação de variáveis (`VariableElimination`).

**Tecnologias:** Python 3, `pgmpy`, `matplotlib`, `reportlab`.

**Como executar:**
```bash
cd rede_bayesiana
pip install -r requirements.txt
python rede_bayesiana_report.py
```

---

### Rede Bayesiana com Interface Streamlit
**Pasta:** [`rede_bayesiana_streamlit/Rede_Bayesiana_Streamlit/`](rede_bayesiana_streamlit/Rede_Bayesiana_Streamlit/)

Versão interativa (web) do mesmo domínio de diagnóstico pulmonar, com dashboard em **Streamlit**: entrada de evidências por formulário, visualização da topologia da rede (NetworkX), inspeção das tabelas de probabilidade condicional (CPDs) de cada nó e geração de laudo em PDF com o resultado da inferência.

**Tecnologias:** Python 3.10+, `streamlit`, `pgmpy`, `pandas`, `networkx`, `matplotlib`, `fpdf2`.

**Como executar:**
```bash
cd rede_bayesiana_streamlit/Rede_Bayesiana_Streamlit
pip install -r requirements.txt
streamlit run rede.py
```

---

### Modelo de Classificação de Peixes (PyTorch/ResNet)
**Pasta:** [`Modelo_ia/`](Modelo_ia/)

Treinamento de uma **ResNet18** pré-treinada, com a camada final adaptada, para classificar imagens de peixes ornamentais em três classes (`acara-bandeira`, `carpa`, `platy-laranja`). Inclui divisão treino/validação/teste, loop de treinamento por 10 épocas e logging de perda/acurácia via **TensorBoard**. O dataset de imagens já está incluído no repositório.

**Tecnologias:** Python 3, `torch`, `torchvision`, `scikit-learn`, `tensorboard`.

**Como executar:**
```bash
cd Modelo_ia
pip install -r requirements.txt
python Modelo_ia.py
```

---

### Exemplo de IA (template de classificação com PyTorch)
**Pasta:** [`ExemplodeIA/`](ExemplodeIA/)

Script-template para um projeto de classificação de imagens com PyTorch/ResNet: define hiperparâmetros, transformações de imagem e o carregamento dos datasets de treino/teste, além da separação treino/validação com `scikit-learn`. Diferente do `Modelo_ia`, este script não inclui um dataset nem executa o treinamento — serve como ponto de partida.

**Tecnologias:** Python 3, `torch`, `torchvision`, `scikit-learn`.

**Como executar:**
```bash
cd ExemplodeIA
pip install -r requirements.txt
python ExemploModeloDeIA.py
```
*(espera uma pasta `./data/train` e `./data/test` no formato `ImageFolder`, que não está incluída neste projeto).*

---

## Tecnologias

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

**Principais bibliotecas:** `pgmpy`, `streamlit`, `torch`, `torchvision`, `scikit-learn`, `networkx`, `matplotlib`, `reportlab`, `fpdf2`.

---

## Como clonar

```bash
git clone https://github.com/pedrogaleano15/IA.git
cd IA
```

Depois, entre na pasta do projeto desejado e instale as dependências específicas dele (veja "Como executar" em cada seção acima).

---

## O que aprendi

- Como modelar incerteza com redes probabilísticas
- Implementação de algoritmos de busca clássicos do zero
- Construção de interfaces interativas com Streamlit
- Diferença prática entre busca cega (BFS/DFS/IDS) e o custo de cada abordagem
- Treinamento e ajuste fino de redes convolucionais (transfer learning com ResNet)

---

## Autor

**Pedro Henrique Morais Galeano**  
Engenharia da Computação · UCDB · Campo Grande/MS  
[GitHub](https://github.com/pedrogaleano15) · [LinkedIn](www.linkedin.com/in/pedro-henrique-morais-galeano)
