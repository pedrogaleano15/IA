# rede_bayesiana_report.py
"""
Rede Bayesiana interativa com menu para:
1. Usar exemplo padrão.
2. Inserir respostas do usuário (e salvar em answers.json).
3. Gerar relatório (por número ou nome).
4. Listar respostas salvas.
Gera PDFs com resultados e gráficos automaticamente.
"""
#bibliorecas necessárias:
# pip install pgmpy reportlab matplotlib


import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.styles import getSampleStyleSheet


# ====================================================
# 1. Construir o modelo da rede Bayesiana
# ====================================================
def build_model():
    model = DiscreteBayesianNetwork([
        ('Poluicao', 'CancerPulmao'),
        ('Fumante', 'CancerPulmao'),
        ('CancerPulmao', 'Dispneia'),
        ('CancerPulmao', 'TossePersistente'),
        ('Tabagismo', 'Fumante'),
        ('Ansiedade', 'Fumante'),
        ('Fumante', 'Enfisema'),
        ('Enfisema', 'Dispneia')
    ])

    cpd_poluicao = TabularCPD('Poluicao', 2, [[0.9], [0.1]])
    cpd_tabagismo = TabularCPD('Tabagismo', 2, [[0.7], [0.3]])
    cpd_ansiedade = TabularCPD('Ansiedade', 2, [[0.8], [0.2]])

    cpd_fumante = TabularCPD(
        'Fumante', 2,
        [[0.9, 0.5, 0.4, 0.05],
         [0.1, 0.5, 0.6, 0.95]],
        evidence=['Tabagismo', 'Ansiedade'],
        evidence_card=[2, 2]
    )

    cpd_cancer = TabularCPD(
        'CancerPulmao', 2,
        [[0.9, 0.8, 0.7, 0.1],
         [0.1, 0.2, 0.3, 0.9]],
        evidence=['Poluicao', 'Fumante'],
        evidence_card=[2, 2]
    )

    cpd_dispneia = TabularCPD(
        'Dispneia', 2,
        [[0.9, 0.3, 0.8, 0.1],
         [0.1, 0.7, 0.2, 0.9]],
        evidence=['CancerPulmao', 'Enfisema'],
        evidence_card=[2, 2]
    )

    cpd_tosse = TabularCPD(
        'TossePersistente', 2,
        [[0.9, 0.2],
         [0.1, 0.8]],
        evidence=['CancerPulmao'],
        evidence_card=[2]
    )

    cpd_enfisema = TabularCPD(
        'Enfisema', 2,
        [[0.95, 0.2],
         [0.05, 0.8]],
        evidence=['Fumante'],
        evidence_card=[2]
    )

    model.add_cpds(cpd_poluicao, cpd_tabagismo, cpd_ansiedade,
                   cpd_fumante, cpd_cancer, cpd_dispneia, cpd_tosse, cpd_enfisema)
    model.check_model()
    return model


# ====================================================
# 2. Funções utilitárias (JSON)
# ====================================================
def load_answers(filename="answers.json"):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {"cenarios": []}


def save_answers(data, filename="answers.json"):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ====================================================
# 3. Menu principal
# ====================================================
def menu():
    while True:
        print("\n=== MENU - Rede Bayesiana ===")
        print("1. Usar exemplo padrão")
        print("2. Inserir nova resposta (e salvar)")
        print("3. Gerar relatório")
        print("4. Listar respostas salvas")
        print("5. Sair")
        opcao = input("Escolha uma opção: ")

        if opcao == '1':
            usar_exemplo()
        elif opcao == '2':
            inserir_resposta()
        elif opcao == '3':
            gerar_relatorio()
        elif opcao == '4':
            listar_respostas()
        elif opcao == '5':
            print("Saindo...")
            break
        else:
            print("Opção inválida.")


# ====================================================
# 4. Usar exemplo padrão
# ====================================================
def usar_exemplo():
    evidence = {"Poluicao": 0, "Tabagismo": 0, "Ansiedade": 0}
    gerar_relatorio(evidence, nome_relatorio="exemplo")


# ====================================================
# 5. Inserir resposta manualmente
# ====================================================
def inserir_resposta():
    print("\n=== Inserir Nova Resposta ===")
    print("Digite 0 (Não) ou 1 (Sim) para cada variável.\n")

    variaveis = ["Poluicao", "Tabagismo", "Ansiedade", "Fumante",
                 "TossePersistente", "Enfisema", "Dispneia"]

    respostas = {}
    for v in variaveis:
        val = input(f"{v} (0=Não, 1=Sim ou Enter p/ deixar em branco): ")
        if val.strip() == "":
            continue
        elif val in ['0', '1']:
            respostas[v] = int(val)
        else:
            print("Valor inválido, ignorando.")

    nome = input("Digite um nome para esse cenário: ").strip()
    if not nome:
        nome = f"cenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    dados = load_answers()
    dados["cenarios"].append({"nome": nome, "respostas": respostas})
    save_answers(dados)
    print(f"[OK] Resposta salva como '{nome}'.")


# ====================================================
# 6. Gerar relatório (por número ou nome)
# ====================================================
def gerar_relatorio(evidence=None, nome_relatorio=None):
    model = build_model()
    infer = VariableElimination(model)

    if evidence is None:
        dados = load_answers()
        if not dados["cenarios"]:
            print("Nenhuma resposta salva. Insira uma antes de gerar relatório.")
            return

        print("\n=== Gerar Relatório ===")
        for i, c in enumerate(dados["cenarios"]):
            print(f"{i+1}. {c['nome']}")
        escolha = input("Escolha um cenário (número ou nome) ou ENTER para gerar todos: ").strip()

        if escolha == "":
            for c in dados["cenarios"]:
                gerar_relatorio(c["respostas"], nome_relatorio=c["nome"])
            return
        else:
            if escolha.isdigit():
                idx = int(escolha) - 1
                if idx < 0 or idx >= len(dados["cenarios"]):
                    print("Índice inválido.")
                    return
                selecionado = dados["cenarios"][idx]
            else:
                selecionado = next((c for c in dados["cenarios"] if c["nome"] == escolha), None)
                if not selecionado:
                    print("Cenário não encontrado.")
                    return

            evidence = selecionado["respostas"]
            nome_relatorio = selecionado["nome"]

    # Consultas de interesse
    variaveis_consulta = ['CancerPulmao', 'Fumante', 'Dispneia', 'Enfisema']
    queries = {}

    for var in variaveis_consulta:
        # evita conflito entre variável e evidência
        if var in evidence:
            print(f"[INFO] Ignorando consulta de {var}, pois já foi definida nas evidências.")
            continue
        resultado = infer.query(variables=[var], evidence=evidence)
        queries[f'P({var})'] = resultado.values

    # gerar gráficos
    figs = []
    os.makedirs("figs", exist_ok=True)
    for q, dist in queries.items():
        plt.figure()
        plt.bar(['0', '1'], dist)
        plt.ylim(0, 1)
        plt.title(q)
        path = f"figs/{nome_relatorio}_{q.replace('/', '_')}.png"
        plt.savefig(path)
        plt.close()
        figs.append(path)

    gerar_pdf(evidence, queries, figs, nome_relatorio)
    print(f"[OK] Relatório '{nome_relatorio}' gerado com sucesso!")


# ====================================================
# 7. Listar respostas salvas
# ====================================================
def listar_respostas():
    dados = load_answers()
    if not dados["cenarios"]:
        print("Nenhuma resposta salva.")
        return
    print("\n=== CENÁRIOS SALVOS ===")
    for i, c in enumerate(dados["cenarios"]):
        print(f"{i+1}. {c['nome']} -> {c['respostas']}")


# ====================================================
# 8. Gerar PDF com resultados
# ====================================================
def gerar_pdf(evidence, queries, figs, nome_relatorio="relatorio"):
    doc = SimpleDocTemplate(f"{nome_relatorio}.pdf", pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"Relatório: {nome_relatorio}", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Evidências fornecidas:", styles['Heading2']))
    for k, v in evidence.items():
        story.append(Paragraph(f"{k}: {v}", styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Resultados das Consultas:", styles['Heading2']))
    for q, dist in queries.items():
        story.append(Paragraph(f"{q}: P(0)={dist[0]:.3f}, P(1)={dist[1]:.3f}", styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Gráficos:", styles['Heading2']))
    for p in figs:
        story.append(Image(p, width=150*mm, height=90*mm))
        story.append(Spacer(1, 6))

    doc.build(story)


# ====================================================
# EXECUÇÃO
# ====================================================
if __name__ == "__main__":
    menu()
