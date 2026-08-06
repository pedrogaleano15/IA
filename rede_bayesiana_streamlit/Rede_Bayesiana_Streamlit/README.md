# 🫁 Sistema de Apoio ao Diagnóstico: Câncer de Pulmão

Este projeto implementa um sistema de inteligência artificial baseado em **Redes Bayesianas** para auxiliar no diagnóstico de doenças pulmonares. Ele modela a incerteza de diagnósticos médicos relacionando fatores de risco (como poluição e tabagismo) com patologias (Câncer e Enfisema) e sintomas observáveis.

📺 **Assista à Demonstração do Projeto:**
[![Assista ao Vídeo](https://img.youtube.com/vi/_jU3nUyWgNg/0.jpg)](https://youtu.be/_jU3nUyWgNg)
*(Clique na imagem para assistir)*

---

## 🚀 Começando

Essas instruções permitirão que você obtenha uma cópia do projeto em operação na sua máquina local para fins de desenvolvimento e teste.

Consulte **Implantação** para saber como implantar o projeto.

### 📋 Pré-requisitos

Você precisa ter o **Python** instalado (versão 3.10 ou superior) e o gerenciador de pacotes `pip`.

Exemplo de verificação no terminal:
```bash
python --version
pip --version
```

### 🔧 Instalação

Uma série de exemplos passo-a-passo que informam o que você deve executar para ter um ambiente de desenvolvimento em execução.

**Passo 1:** Clone o repositório para sua máquina local.
```bash
git clone https://github.com/pedrogaleano15/IA.git
cd IA/rede_bayesiana_streamlit/Rede_Bayesiana_Streamlit
```

**Passo 2:** Instale as dependências listadas no projeto.
```bash
pip install -r requirements.txt
```

**Passo 3:** Execute a aplicação para iniciar o servidor local.
```bash
streamlit run rede.py
```

**Passo 4:** O sistema estará acessível no seu navegador em:
```
http://localhost:8501
```

## ⚙️ Executando os testes

Para validar o funcionamento do sistema, utilize a interface gráfica na aba "Anamnese & Diagnóstico".

### 🔩 Testes de ponta a ponta

Estes testes verificam a lógica probabilística da rede bayesiana (inferência).

**Exemplo 1: Teste de Risco Elevado**
- Ação: Na interface, defina Fumante = Sim e Poluição = Alta.
- Resultado: A probabilidade de Câncer de Pulmão deve subir para ~30%.
- Porquê: Verifica se a rede propaga corretamente o risco cumulativo.

**Exemplo 2: Teste de "Explaining Away"**
- Ação: Defina Dispneia = Presente. Depois, adicione Enfisema = Sim.
- Resultado: A probabilidade de Câncer deve cair levemente (de ~15% para ~10%).
- Porquê: Verifica o raciocínio abdutivo (o Enfisema já explica o sintoma, reduzindo a necessidade de outra causa).

### ⌨️ Estilo de codificação

O código segue as diretrizes da PEP 8 para Python.

- **Docstrings:** as funções principais (`definir_modelo`, `criar_grafico_rede`) possuem documentação interna explicativa.
- **Tipagem:** uso implícito de tipos compatíveis com `pgmpy` e `pandas`.

## 📦 Implantação

Para implantar em um sistema ativo (como Streamlit Cloud, Heroku ou AWS):

1. Certifique-se de que o `requirements.txt` está atualizado (`pip freeze > requirements.txt`).
2. Configure o servidor para expor a porta 8501 (padrão do Streamlit).

## 🛠️ Construído com

Ferramentas utilizadas para criar o projeto:

- [Streamlit](https://streamlit.io/) - O framework web para a interface gráfica
- [pgmpy](https://pgmpy.org/) - Biblioteca para modelagem e inferência Bayesiana
- [NetworkX](https://networkx.org/) - Geração de gráficos e topologia de redes
- [FPDF2](https://py-pdf.github.io/fpdf2/) - Biblioteca para geração de relatórios em PDF

## 📌 Versão

Nós usamos SemVer para controle de versão. Para as versões disponíveis, observe as tags neste repositório.

## ✒️ Autores

- **Pedro Henrique Morais Galeano** - Desenvolvimento e Modelagem
- **Mateus Soltoky** - Desenvolvimento e Documentação
