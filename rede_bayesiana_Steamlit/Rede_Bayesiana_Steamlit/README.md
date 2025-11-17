# 🫁 Sistema de Apoio ao Diagnóstico Médico via Redes Bayesianas

**Disciplina:** Inteligência Artificial / Sistemas Distribuídos  
**Autores:** * Pedro Henrique Morais Galeano
* Mateus Soltoky

## 📋 Sobre o Projeto
Este software é um Sistema de Apoio à Decisão Clínica (CDSS) desenvolvido em Python. Ele utiliza **Redes Bayesianas** para calcular a probabilidade de doenças pulmonares (Câncer de Pulmão e Enfisema) com base em fatores de risco e sintomas observados.

A aplicação oferece uma interface gráfica para anamnese, visualização da rede causal e geração automática de laudos médicos em PDF.

## 📂 Estrutura de Arquivos
A organização do projeto é a seguinte:

```text
bayes-med-diagnostic/
├── rede.py                      # Código fonte principal (Aplicação Streamlit)
├── relatorios_pdf/              # Diretório gerado automaticamente para saídas
│   ├── temp_network_graph.png   # Imagem temporária da topologia da rede
│   └── relatorio_medico_*.pdf   # Laudos gerados com timestamp
├── requirements.txt             # Lista de dependências
└── README.md                    # Documentação do projeto

🧠 O Modelo Probabilístico
A rede implementada em rede.py possui 8 variáveis interconectadas:

1. Fatores de Risco (Raízes)
Poluição: Nível de exposição ambiental.

Tabagismo (Histórico): Histórico familiar ou social.

Ansiedade: Fator psicológico influenciador.

2. Condições (Nós Intermediários)
Fumante: Hábito ativo (Depende de Ansiedade e Histórico).

Enfisema: Doença crônica (Depende de Fumante).

Câncer de Pulmão: Patologia grave (Depende de Fumante e Poluição).

3. Sintomas (Folhas)
Tosse Persistente: Indicativo principal de Câncer.

Dispneia (Falta de Ar): Indicativo de Enfisema ou Câncer.

🚀 Como Executar
Pré-requisitos
Certifique-se de ter o Python 3.10+ instalado.

Instale as bibliotecas necessárias:

Bash

pip install streamlit pgmpy pandas networkx matplotlib fpdf numpy
Execute a aplicação:

Bash

streamlit run rede.py
Acesse: O navegador abrirá automaticamente em http://localhost:8501.

🛠️ Tecnologias
Frontend: Streamlit

Inferência: pgmpy (Variable Elimination)

Visualização: NetworkX & Matplotlib

Relatórios: FPDF