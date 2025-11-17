import streamlit as st
import pandas as pd
import os
import time
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from fpdf import FPDF
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# --- Configuração de Pastas ---
# Cria o diretório para salvar os relatórios PDF caso não exista
PDF_OUTPUT_DIR = "relatorios_pdf"
GRAPH_IMG_FILENAME = "temp_network_graph.png"
os.makedirs(PDF_OUTPUT_DIR, exist_ok=True)

# --- Função para Gerar o Gráfico ---
def criar_grafico_rede(model):
    """
    Gera uma imagem visual da topologia da rede Bayesiana.
    Usa a biblioteca NetworkX para desenhar os nós e arestas.
    """
    try:
        G = nx.DiGraph()
        G.add_edges_from(model.edges())
        
        # Tenta um layout planar (melhor organização), se falhar usa spring
        try:
            pos = nx.planar_layout(G)
        except:
            pos = nx.spring_layout(G, seed=42, k=3) 
            
        plt.figure(figsize=(14, 10))
        # Desenha os nós (variáveis)
        nx.draw_networkx_nodes(G, pos, node_size=3500, node_color='#ffcccc', edgecolors='#cc0000', linewidths=2)
        # Desenha as arestas (relações de causa)
        nx.draw_networkx_edges(G, pos, edge_color='#666666', arrows=True, arrowsize=20, width=1.5)
        # Adiciona os rótulos
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', font_family='sans-serif')
        
        plt.title("Rede Bayesiana - Diagnóstico Pulmonar", fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        
        # Salva a imagem para ser usada no PDF e na interface
        graph_image_path = os.path.join(PDF_OUTPUT_DIR, GRAPH_IMG_FILENAME)
        plt.savefig(graph_image_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        return graph_image_path
    except Exception as e:
        st.error(f"Erro ao criar gráfico: {e}")
        return None

# --- Funções de PDF ---
def criar_pdf_relatorio():
    """
    Gera um relatório PDF completo com os dados da sessão atual (gráficos, evidências e resultados).
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Cabeçalho
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Relatório de Diagnóstico Médico (IA)", ln=True, align='C')
    pdf.ln(5)
    
    # Data e Hora
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, f"Gerado em: {time.strftime('%d/%m/%Y %H:%M:%S')}", ln=True, align='C')
    pdf.ln(10)

    # 1. Inclusão da Imagem da Rede
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "1. Topologia da Rede (Fatores de Risco)", ln=True)
    if 'graph_image_path' in st.session_state and os.path.exists(st.session_state['graph_image_path']):
        try:
            pdf.image(st.session_state['graph_image_path'], x=10, w=190)
        except:
            pdf.cell(0, 10, "(Imagem não disponível)", ln=True)
    pdf.ln(10)

    # 2. Listagem das Evidências (Sintomas informados)
    if 'evidencias_para_pdf' in st.session_state:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "2. Anamnese (Sintomas e Histórico)", ln=True)
        pdf.set_font("Arial", size=10)
        evidencias = st.session_state['evidencias_para_pdf']
        if evidencias:
            for k, v in evidencias.items():
                pdf.cell(0, 6, f"- {k}: {v}", ln=True)
        else:
            pdf.cell(0, 6, "Nenhuma evidência definida.", ln=True)
        pdf.ln(10)

    # 3. Resultados da Inferência
    if 'resultados_para_pdf' in st.session_state:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "3. Diagnóstico Probabilístico", ln=True)
        
        for var, df_res in st.session_state['resultados_para_pdf'].items():
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, f"Condição Investigada: {var}", ln=True)
            
            pdf.set_font("Courier", size=9)
            if isinstance(df_res, pd.DataFrame):
                for idx, row in df_res.iterrows():
                    val = row['Probabilidade'] if 'Probabilidade' in row else row.iloc[0]
                    texto = f"  {str(idx).ljust(15)}: {val:.4f} ({val*100:5.1f}%)"
                    pdf.cell(0, 5, texto, ln=True)
            pdf.ln(5)

    filename = f"relatorio_medico_{int(time.time())}.pdf"
    filepath = os.path.join(PDF_OUTPUT_DIR, filename)
    try:
        pdf.output(filepath)
        return filepath
    except Exception as e:
        return f"Erro ao salvar PDF: {str(e)}"

# --- Definição do Modelo (Topologia e Probabilidades) ---
def definir_modelo():
    """
    Define a estrutura da Rede Bayesiana e suas Tabelas de Probabilidade Condicional (CPDs).
    """
    # Definindo as arestas (quem influencia quem)
    model = DiscreteBayesianNetwork([
        ('Poluicao', 'CancerPulmao'),       # Poluição aumenta risco de Câncer
        ('Fumante', 'CancerPulmao'),        # Fumar aumenta risco de Câncer
        ('CancerPulmao', 'Dispneia'),       # Câncer causa falta de ar
        ('CancerPulmao', 'TossePersistente'), # Câncer causa tosse
        ('Tabagismo', 'Fumante'),           # Histórico influencia hábito atual
        ('Ansiedade', 'Fumante'),           # Ansiedade influencia hábito atual
        ('Fumante', 'Enfisema'),            # Fumar causa Enfisema
        ('Enfisema', 'Dispneia')            # Enfisema causa falta de ar
    ])

    # --- Definição das CPDs (Valores ilustrativos baseados em lógica médica) ---

    # 1. Nós Raízes (Sem pais)
    cpd_poluicao = TabularCPD('Poluicao', 2, [[0.8], [0.2]], state_names={'Poluicao': ['Baixa', 'Alta']})
    cpd_tabagismo = TabularCPD('Tabagismo', 2, [[0.7], [0.3]], state_names={'Tabagismo': ['Nao', 'Sim']}) 
    cpd_ansiedade = TabularCPD('Ansiedade', 2, [[0.6], [0.4]], state_names={'Ansiedade': ['Baixa', 'Alta']})

    # 2. Nós Intermediários (Dependem dos pais)
    # A probabilidade de ser Fumante depende de Histórico (Tabagismo) e Ansiedade
    cpd_fumante = TabularCPD('Fumante', 2, 
                             [[0.95, 0.8, 0.6, 0.1],  # Nao Fuma
                              [0.05, 0.2, 0.4, 0.9]], # Fuma
                             evidence=['Tabagismo', 'Ansiedade'], evidence_card=[2, 2],
                             state_names={'Fumante': ['Nao', 'Sim'], 'Tabagismo': ['Nao', 'Sim'], 'Ansiedade': ['Baixa', 'Alta']})

    # Enfisema depende apenas se a pessoa fuma
    cpd_enfisema = TabularCPD('Enfisema', 2, 
                              [[0.98, 0.3],  # Nao
                               [0.02, 0.7]], # Sim (Alto risco para fumantes)
                              evidence=['Fumante'], evidence_card=[2],
                              state_names={'Enfisema': ['Nao', 'Sim'], 'Fumante': ['Nao', 'Sim']})

    # Câncer depende de Poluição e se a pessoa Fuma
    cpd_cancer = TabularCPD('CancerPulmao', 2,
                            [[0.99, 0.9, 0.95, 0.7],  # Negativo
                             [0.01, 0.1, 0.05, 0.3]], # Positivo (Risco cumulativo)
                            evidence=['Poluicao', 'Fumante'], evidence_card=[2, 2],
                            state_names={'CancerPulmao': ['Negativo', 'Positivo'], 'Poluicao': ['Baixa', 'Alta'], 'Fumante': ['Nao', 'Sim']})

    # 3. Nós Folhas (Sintomas)
    cpd_tosse = TabularCPD('TossePersistente', 2,
                           [[0.9, 0.2],  # Ausente
                            [0.1, 0.8]], # Presente (Muito comum no câncer)
                           evidence=['CancerPulmao'], evidence_card=[2],
                           state_names={'TossePersistente': ['Ausente', 'Presente'], 'CancerPulmao': ['Negativo', 'Positivo']})

    cpd_dispneia = TabularCPD('Dispneia', 2,
                              [[0.9, 0.4, 0.3, 0.1],  # Ausente
                               [0.1, 0.6, 0.7, 0.9]], # Presente (Causado por ambas as doenças)
                              evidence=['CancerPulmao', 'Enfisema'], evidence_card=[2, 2],
                              state_names={'Dispneia': ['Ausente', 'Presente'], 'CancerPulmao': ['Negativo', 'Positivo'], 'Enfisema': ['Nao', 'Sim']})

    # Adiciona todas as CPDs ao modelo
    model.add_cpds(cpd_poluicao, cpd_tabagismo, cpd_ansiedade, cpd_fumante, cpd_enfisema, cpd_cancer, cpd_tosse, cpd_dispneia)
    
    # Verificação de integridade do modelo
    if not model.check_model():
        raise ValueError("Erro na definição das CPDs. Verifique as dimensões.")
        
    return model

# --- Lógica da Interface (Streamlit) ---
st.set_page_config(page_title="Diagnóstico Pulmonar Bayesiano", layout="wide", page_icon="🫁")
st.title("🫁 Sistema de Apoio ao Diagnóstico: Câncer de Pulmão")

try:
    # Inicializa modelo e inferência
    model = definir_modelo()
    inference = VariableElimination(model)
    
    # Gera o gráfico apenas uma vez para otimizar
    if 'graph_created' not in st.session_state:
        with st.spinner("Carregando modelo clínico..."):
            path = criar_grafico_rede(model)
            if path: st.session_state['graph_image_path'] = path
            st.session_state['graph_created'] = True

    # Interface em Abas
    tab1, tab2, tab3 = st.tabs(["📋 Anamnese & Diagnóstico", "📊 Estrutura da Rede", "📄 Laudo Médico"])

    # ABA 1: Entrada de Dados e Cálculo
    with tab1:
        st.markdown("### Dados do Paciente e Sintomas")
        col_ev, col_res = st.columns([1, 2])
        
        with col_ev:
            st.subheader("1. Entradas (Evidências)")
            st.info("Insira o histórico e sintomas observados:")
            
            evidence_dict = {}
            
            # Seleção de Fatores de Risco
            with st.expander("Fatores de Risco & Histórico", expanded=True):
                for var in ['Tabagismo', 'Ansiedade', 'Poluicao', 'Fumante']:
                    states = model.get_cpds(var).state_names[var]
                    val = st.selectbox(f"{var}", ['Desconhecido'] + states)
                    if val != 'Desconhecido': evidence_dict[var] = val
            
            # Seleção de Sintomas
            with st.expander("Sintomas Clínicos"):
                for var in ['TossePersistente', 'Dispneia']:
                    states = model.get_cpds(var).state_names[var]
                    val = st.selectbox(f"{var}", ['Desconhecido'] + states)
                    if val != 'Desconhecido': evidence_dict[var] = val

        with col_res:
            st.subheader("2. Investigação Clínica")
            target_var = st.selectbox("Qual condição investigar?", 
                                    ['CancerPulmao', 'Enfisema', 'Fumante', 'Ansiedade'], 
                                    index=0)
            
            if st.button("🩺 Calcular Probabilidade", type="primary"):
                try:
                    # Remove a variável alvo das evidências para evitar erro circular
                    final_evidence = evidence_dict.copy()
                    if target_var in final_evidence:
                        del final_evidence[target_var]
                        st.caption(f"ℹ️ Nota: Ignorando entrada manual de '{target_var}' para realizar o cálculo.")

                    st.session_state['evidencias_para_pdf'] = final_evidence
                    
                    # Realiza a inferência exata
                    with st.spinner("Processando inferência bayesiana..."):
                        result = inference.query(variables=[target_var], evidence=final_evidence)
                        
                    states = result.state_names[target_var]
                    values = result.values.flatten()
                    
                    # Prepara DataFrame para exibição
                    df = pd.DataFrame(values, index=states, columns=['Probabilidade'])
                    df.index.name = 'Condição'
                    
                    # Exibe gráfico e tabela
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.bar_chart(df, color="#ff4b4b")
                    with c2:
                        st.dataframe(df.style.format("{:.2%}"))
                    
                    # Alerta inteligente baseado na probabilidade
                    top_state = df['Probabilidade'].idxmax()
                    prob = df['Probabilidade'].max()
                    
                    if top_state in ['Sim', 'Positivo', 'Presente', 'Alta'] and prob > 0.7:
                        st.warning(f"⚠️ Alerta: Alta probabilidade de **{target_var}** estar **{top_state}** ({prob:.1%})")
                    else:
                        st.success(f"Resultado: **{target_var}** tende a **{top_state}** ({prob:.1%})")
                    
                    # Salva resultado para o PDF
                    st.session_state['resultados_para_pdf'] = {target_var: df}
                    
                except Exception as e:
                    st.error(f"Erro na inferência: {e}")

    # ABA 2: Visualização da Rede
    with tab2:
        col_g, col_c = st.columns(2)
        with col_g:
            if 'graph_image_path' in st.session_state:
                st.image(st.session_state['graph_image_path'], caption="Grafo de Dependência (Causas → Efeitos)")
        with col_c:
            st.subheader("Probabilidades Condicionais (CPDs)")
            node_sel = st.selectbox("Visualizar Nó", sorted(model.nodes()))
            cpd = model.get_cpds(node_sel)
            
            # Formata a tabela para visualização correta em 2D
            vals = cpd.values
            vals_2d = vals.reshape(cpd.variable_card, -1)
            
            st.write(f"Nó: **{node_sel}**")
            evidence_vars = cpd.variables[1:] if len(cpd.variables) > 1 else []
            
            if evidence_vars:
                st.info(f"Depende de: {', '.join(evidence_vars)}")
            else:
                st.success("Nó Raiz (Independente)")
            
            df_display = pd.DataFrame(vals_2d)
            st.dataframe(df_display.style.format("{:.2f}"))
            st.caption("Linhas = Estados da Variável | Colunas = Combinações dos Pais")
            
            st.session_state['cpd_para_pdf'] = df_display 
            st.session_state['cpd_node'] = node_sel

    # ABA 3: Relatório
    with tab3:
        st.header("Emitir Laudo / Relatório")
        st.markdown("Gera um arquivo PDF com o resumo da anamnese, o gráfico da rede e o resultado probabilístico.")
        if st.button("📄 Baixar Laudo em PDF"):
            if 'resultados_para_pdf' not in st.session_state:
                st.warning("Realize uma investigação clínica na primeira aba antes de gerar o laudo.")
            else:
                path = criar_pdf_relatorio()
                if "Erro" in path:
                    st.error(path)
                else:
                    with open(path, "rb") as f:
                        st.download_button("⬇️ Download Laudo", f, file_name="laudo_medico_bayes.pdf")

except Exception as e:
    st.error(f"Erro crítico na inicialização: {e}")