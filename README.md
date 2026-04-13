# Inteligência Artificial — Estudos e Projetos Práticos

Repositório com implementações práticas de conceitos fundamentais de Inteligência Artificial, desenvolvidos durante a graduação em Engenharia da Computação na UCDB.

---

## Conteúdo

### BFS — Busca em Largura
Implementação do algoritmo de **Busca em Largura (Breadth-First Search)** aplicado à resolução de problemas de busca em espaço de estados. O algoritmo explora sistematicamente todos os nós de um grafo por nível, garantindo encontrar a solução de menor custo em grafos não ponderados.

**Conceitos aplicados:** grafos, fila (FIFO), complexidade de tempo O(V + E).

---

### Rede Bayesiana
Modelagem e implementação de uma **Rede Bayesiana** para inferência probabilística. O projeto permite calcular probabilidades condicionais com base em evidências observadas, demonstrando o raciocínio sob incerteza.

**Conceitos aplicados:** probabilidade condicional, teorema de Bayes, independência condicional, inferência por variável de eliminação.

---

### Rede Bayesiana com Interface Streamlit
Versão interativa da Rede Bayesiana com interface web construída com **Streamlit**, permitindo ao usuário inserir evidências e visualizar as probabilidades inferidas em tempo real.

**Como executar:**
```bash
pip install streamlit pgmpy
streamlit run rede_bayesiana_Steamlit/Rede_Bayesiana_Steamlit/app.py
```

---

### Modelo de IA / Exemplo de IA
Experimentos adicionais com modelos de IA, explorando conceitos como classificação e aprendizado supervisionado.

---

## Tecnologias

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

**Bibliotecas:** `pgmpy`, `streamlit`, `networkx`

---

## Como clonar e rodar

```bash
git clone https://github.com/pedrogaleano15/IA.git
cd IA
pip install -r requirements.txt
```

---

## O que aprendi

- Como modelar incerteza com redes probabilísticas
- Implementação de algoritmos de busca clássicos do zero
- Construção de interfaces interativas com Streamlit
- Diferença prática entre busca cega (BFS) e busca informada

---

## Autor

**Pedro Henrique Morais Galeano**  
Engenharia da Computação · UCDB · Campo Grande/MS  
[GitHub](https://github.com/pedrogaleano15) · [LinkedIn](www.linkedin.com/in/pedro-henrique-morais-galeano)
