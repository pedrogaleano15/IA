# Rede Bayesiana Interativa com Geração de Relatórios

Este projeto implementa uma Rede Bayesiana simples (usando `pgmpy`) focada em um cenário de fatores de risco para Câncer de Pulmão. Ele oferece um menu interativo para realizar inferências e gera automaticamente relatórios em PDF com os resultados e gráficos.

## 📊 Funcionalidades

* **Modelo Bayesiano:** Define uma rede com variáveis como Poluição, Tabagismo, Fumante, Câncer de Pulmão, Enfisema, Dispneia, etc., e suas probabilidades condicionais (CPDs).
* **Menu Interativo:** Permite ao usuário:
    * Rodar um cenário de exemplo.
    * Inserir suas próprias evidências (respostas sobre as variáveis) e salvá-las em `answers.json`.
    * Gerar relatórios em PDF para cenários salvos (por número ou nome) ou para o exemplo padrão.
    * Listar os cenários já salvos no arquivo `answers.json`.
* **Inferência:** Utiliza `VariableElimination` do `pgmpy` para calcular as probabilidades marginais das variáveis de interesse (Câncer, Fumante, etc.) dadas as evidências.
* **Geração de Gráficos:** Cria gráficos de barra (`matplotlib`) para visualizar as distribuições de probabilidade resultantes e os salva na pasta `figs/`.
* **Geração de Relatórios:** Usa `reportlab` para compilar as evidências, os resultados das inferências e os gráficos em um arquivo PDF nomeado de acordo com o cenário.

## 🛠️ Tecnologias Utilizadas

* **Python 3**
* **pgmpy:** Para modelagem e inferência em redes bayesianas.
* **reportlab:** Para geração de arquivos PDF.
* **matplotlib:** Para criação de gráficos.
* **json:** Para salvar e carregar cenários do usuário.

## 📦 Instalação

1.  **Clone o repositório e entre na pasta:**
    ```bash
    git clone [https://github.com/pedrogaleano15/GUSTAVO-IA.git](https://github.com/pedrogaleano15/GUSTAVO-IA.git)
    cd GUSTAVO-IA/rede_bayesian
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

## 🏃 Como Executar

Com o ambiente virtual ativado, execute o script principal:

```bash
python rede_bayesiana_report.py