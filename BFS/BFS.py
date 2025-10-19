from collections import deque
import time

# Estado objetivo do 8-puzzle
GOAL = (1, 2, 3,
        4, 5, 6,
        7, 8, 0)

def show(state):
    """Exibe o estado atual do puzzle em formato 3x3"""
    for i in range(0, 9, 3):
        print(state[i], state[i+1], state[i+2])
    print()

def find_zero(state):
    """Encontra a posição do zero (espaço vazio) no estado"""
    return state.index(0)

def get_moves(z_pos):
    """Retorna os movimentos possíveis a partir da posição do zero"""
    moves = []
    row, col = divmod(z_pos, 3)
    if row > 0: moves.append('up')
    if row < 2: moves.append('down')
    if col > 0: moves.append('left')
    if col < 2: moves.append('right')
    return moves

def apply_move(state, move, z_pos):
    """Aplica um movimento ao estado atual"""
    if move == 'up': new_pos = z_pos - 3
    elif move == 'down': new_pos = z_pos + 3
    elif move == 'left': new_pos = z_pos - 1
    elif move == 'right': new_pos = z_pos + 1
    else: return None
    
    state_list = list(state)
    state_list[z_pos], state_list[new_pos] = state_list[new_pos], state_list[z_pos]
    return tuple(state_list)

def is_solvable(state):
    """Verifica se o estado é solúvel (número par de inversões)"""
    seq = [x for x in state if x != 0]
    inversions = sum(
        1 for i in range(len(seq))
        for j in range(i+1, len(seq))
        if seq[i] > seq[j]
    )
    return inversions % 2 == 0

def bfs_metrics(start):
    """Busca em largura (BFS) com métricas"""
    if not is_solvable(start):
        print("Esse estado inicial é insolúvel!")
        return None, None
    
    if start == GOAL:
        return [start], {"nodes_expanded": 0, "frontier_max": 1, "solution_len": 0, "time": 0.0}
    
    queue = deque([[start]])
    visited = {start}
    nodes_expanded = 0
    frontier_max = 1
    start_time = time.time()
    
    while queue:
        path = queue.popleft()
        current = path[-1]
        nodes_expanded += 1
        z_pos = find_zero(current)
        
        for move in get_moves(z_pos):
            new_state = apply_move(current, move, z_pos)
            if new_state is None or new_state in visited:
                continue
            
            if new_state == GOAL:
                solution = path + [new_state]
                metrics = {
                    "nodes_expanded": nodes_expanded,
                    "frontier_max": frontier_max,
                    "solution_len": len(solution) - 1,
                    "time": time.time() - start_time
                }
                return solution, metrics
            
            visited.add(new_state)
            queue.append(path + [new_state])
            frontier_max = max(frontier_max, len(queue))
    
    return None, {"nodes_expanded": nodes_expanded, "frontier_max": frontier_max, "solution_len": None, "time": time.time() - start_time}

def dfs_metrics(start):
    """Busca em profundidade (DFS) com métricas"""
    start_time = time.time()
    nodes_expanded = 0
    frontier_max = 1
    stack = [[start]]
    visited = {start}
    
    while stack:
        frontier_max = max(frontier_max, len(stack))
        path = stack.pop()
        state = path[-1]
        nodes_expanded += 1
        
        if state == GOAL:
            tempo_total = time.time() - start_time
            return path, {"nodes_expanded": nodes_expanded, "frontier_max": frontier_max, 
                         "solution_len": len(path) - 1, "time": tempo_total}
        
        z = find_zero(state)
        for m in get_moves(z):
            ns = apply_move(state, m, z)
            if ns is not None and ns not in visited:
                visited.add(ns)
                stack.append(path + [ns])
    
    tempo_total = time.time() - start_time
    return None, {"nodes_expanded": nodes_expanded, "frontier_max": frontier_max, 
                 "solution_len": None, "time": tempo_total}

def dls(state, depth, visited, nodes_expanded):
    """Busca em profundidade limitada (DLS) para IDS"""
    if state == GOAL:
        return [state], nodes_expanded + 1
    
    if depth == 0:
        return None, nodes_expanded + 1
    
    visited.add(state)
    z = find_zero(state)
    nodes_expanded += 1
    
    for m in get_moves(z):
        ns = apply_move(state, m, z)
        if ns is not None and ns not in visited:
            result, nodes_expanded = dls(ns, depth - 1, visited, nodes_expanded)
            if result:
                return [state] + result, nodes_expanded
    
    return None, nodes_expanded

def ids_metrics(start):
    """Busca em aprofundamento iterativo (IDS) com métricas"""
    start_time = time.time()
    total_nodes_expanded = 0
    frontier_max = 1
    
    for depth in range(100):  # Limite máximo de profundidade
        visited = set()
        result, nodes_expanded = dls(start, depth, visited, 0)
        total_nodes_expanded += nodes_expanded
        frontier_max = max(frontier_max, len(visited))
        
        if result:
            tempo_total = time.time() - start_time
            return result, {"nodes_expanded": total_nodes_expanded, "frontier_max": frontier_max, 
                           "solution_len": len(result) - 1, "time": tempo_total}
    
    tempo_total = time.time() - start_time
    return None, {"nodes_expanded": total_nodes_expanded, "frontier_max": frontier_max, 
                 "solution_len": None, "time": tempo_total}

def run_comparative_analysis():
    """Executa análise comparativa dos três algoritmos"""
    casos = {
        "Fácil": (1, 2, 3, 4, 5, 6, 7, 0, 8),
        "Médio": (1, 2, 3, 4, 5, 6, 0, 7, 8),
        "Desafiador": (1, 2, 3, 5, 0, 6, 4, 7, 8),
    }
    
    resultados = []
    
    for nome, estado in casos.items():
        print(f"\n{'='*60}")
        print(f"ANALISANDO CASO: {nome}")
        print(f"{'='*60}")
        
        # BFS
        print("Executando BFS...")
        solucao_bfs, metricas_bfs = bfs_metrics(estado)
        if solucao_bfs:
            resultados.append([nome, "BFS", metricas_bfs["solution_len"], metricas_bfs["nodes_expanded"], 
                             metricas_bfs["frontier_max"], metricas_bfs["time"]])
            print(f"BFS: {metricas_bfs['solution_len']} movimentos, {metricas_bfs['nodes_expanded']} nós expandidos")
        
        # DFS
        print("Executando DFS...")
        solucao_dfs, metricas_dfs = dfs_metrics(estado)
        if solucao_dfs:
            resultados.append([nome, "DFS", metricas_dfs["solution_len"], metricas_dfs["nodes_expanded"], 
                             metricas_dfs["frontier_max"], metricas_dfs["time"]])
            print(f"DFS: {metricas_dfs['solution_len']} movimentos, {metricas_dfs['nodes_expanded']} nós expandidos")
        else:
            print("DFS não encontrou solução")
        
        # IDS
        print("Executando IDS...")
        solucao_ids, metricas_ids = ids_metrics(estado)
        if solucao_ids:
            resultados.append([nome, "IDS", metricas_ids["solution_len"], metricas_ids["nodes_expanded"], 
                             metricas_ids["frontier_max"], metricas_ids["time"]])
            print(f"IDS: {metricas_ids['solution_len']} movimentos, {metricas_ids['nodes_expanded']} nós expandidos")
        else:
            print("IDS não encontrou solução")
    
    # Exibir tabela de resultados
    print(f"\n{'='*100}")
    print("TABELA COMPARATIVA - RESULTADOS")
    print(f"{'='*100}")
    print(f"{'Caso':<12} | {'Algoritmo':<10} | {'Profundidade':<12} | {'Expandidos':<12} | {'Fronteira Máx':<14} | {'Tempo (s)':<10}")
    print(f"{'-'*100}")
    
    for resultado in resultados:
        caso, algoritmo, profundidade, expandidos, fronteira, tempo = resultado
        print(f"{caso:<12} | {algoritmo:<10} | {profundidade:<12} | {expandidos:<12} | {fronteira:<14} | {tempo:<10.4f}")

def select_start_state():
    """Menu para selecionar o estado inicial"""
    print("\n" + "="*50)
    print("SELECIONE O ESTADO INICIAL:")
    print("="*50)
    print("1 - Estado fácil:")
    print("   1,2,3,")
    print("   4,5,6,") 
    print("   7,0,8")
    print("\n2 - Estado médio:")
    print("   1,2,3,")
    print("   4,5,6,")
    print("   0,7,8")
    print("\n3 - Estado desafiador:")
    print("   1,2,3,")
    print("   5,0,6,")
    print("   4,7,8")
    print("\n4 - Análise comparativa completa (todos os casos)")
    print("\n5 - Personalizado (digitar manualmente)")
    
    choice = input("\nDigite o número da opção desejada: ")
    
    states = {
        '1': (1, 2, 3, 4, 5, 6, 7, 0, 8),
        '2': (1, 2, 3, 4, 5, 6, 0, 7, 8),
        '3': (1, 2, 3, 5, 0, 6, 4, 7, 8),
    }
    
    if choice in ['1', '2', '3']:
        return states[choice]
    elif choice == '4':
        run_comparative_analysis()
        return None
    elif choice == '5':
        print("\nDigite os 9 números do puzzle (0 representa o espaço vazio):")
        print("Exemplo: 1 2 3 4 5 6 7 8 0")
        try:
            values = list(map(int, input("Digite os valores: ").split()))
            if len(values) != 9:
                print("Erro: deve digitar exatamente 9 números!")
                return select_start_state()
            return tuple(values)
        except ValueError:
            print("Erro: digite apenas números!")
            return select_start_state()
    else:
        print("Opção inválida! Usando estado padrão.")
        return (1, 2, 3, 0, 5, 6, 4, 7, 8)

def select_algorithm():
    """Menu para selecionar o algoritmo de busca"""
    print("\n" + "="*50)
    print("SELECIONE O ALGORITMO DE BUSCA:")
    print("="*50)
    print("1 - BFS (Busca em Largura)")
    print("2 - DFS (Busca em Profundidade)")
    print("3 - IDS (Busca em Aprofundamento Iterativo)")
    print("4 - Voltar ao menu anterior")
    print("5 - Sair do programa")
    
    return input("\nDigite o número da opção desejada: ")

# Programa principal
if __name__ == "__main__":
    while True:
        print("\n" + "="*60)
        print("RESOLVEDOR DE 8-PUZZLE")
        print("="*60)
        
        # Selecionar estado inicial
        start = select_start_state()
        
        if start is None:  # Caso de análise comparativa
            continue
            
        print(f"\nEstado inicial selecionado:")
        show(start)
        
        if not is_solvable(start):
            print("Este estado é INSOLÚVEL!")
            print("O puzzle 8-puzzle requer um número par de inversões para ser solúvel.")
            continue
        
        # Menu de algoritmos
        while True:
            escolha = select_algorithm()
            
            if escolha == '1':
                print("\nExecutando BFS...")
                solution, metrics = bfs_metrics(start)
                if solution:
                    print(f"Solução encontrada em {metrics['solution_len']} movimentos!")
                    mostrar_solucao = input("Deseja ver a solução passo a passo? (s/n): ").lower()
                    if mostrar_solucao == 's':
                        for step, state in enumerate(solution):
                            print(f"Passo {step}:")
                            show(state)
                    print("\nMétricas do BFS:")
                    print(f"- Nós expandidos: {metrics['nodes_expanded']}")
                    print(f"- Tamanho máximo da fronteira: {metrics['frontier_max']}")
                    print(f"- Tempo de execução: {metrics['time']:.4f}s")
                else:
                    print("Nenhuma solução encontrada!")
            
            elif escolha == '2':
                print("\nExecutando DFS...")
                solution, metrics = dfs_metrics(start)
                if solution:
                    print(f"Solução encontrada em {metrics['solution_len']} movimentos!")
                    mostrar_solucao = input("Deseja ver a solução passo a passo? (s/n): ").lower()
                    if mostrar_solucao == 's':
                        for step, state in enumerate(solution):
                            print(f"Passo {step}:")
                            show(state)
                    print("\nMétricas do DFS:")
                    print(f"- Nós expandidos: {metrics['nodes_expanded']}")
                    print(f"- Tamanho máximo da fronteira: {metrics['frontier_max']}")
                    print(f"- Tempo de execução: {metrics['time']:.4f}s")
                else:
                    print("Nenhuma solução encontrada com DFS!")
            
            elif escolha == '3':
                print("\nExecutando IDS...")
                solution, metrics = ids_metrics(start)
                if solution:
                    print(f"Solução encontrada em {metrics['solution_len']} movimentos!")
                    mostrar_solucao = input("Deseja ver a solução passo a passo? (s/n): ").lower()
                    if mostrar_solucao == 's':
                        for step, state in enumerate(solution):
                            print(f"Passo {step}:")
                            show(state)
                    print("\nMétricas do IDS:")
                    print(f"- Nós expandidos: {metrics['nodes_expanded']}")
                    print(f"- Tamanho máximo da fronteira: {metrics['frontier_max']}")
                    print(f"- Tempo de execução: {metrics['time']:.4f}s")
                else:
                    print("Nenhuma solução encontrada com IDS!")
            
            elif escolha == '4':
                print("Voltando ao menu anterior...")
                break
            
            elif escolha == '5':
                print("Saindo do programa. Até logo!")
                exit()
            
            else:
                print("Opção inválida! Tente novamente.")
            
            # Perguntar se quer continuar com o mesmo estado
            continuar = input("\nDeseja testar outro algoritmo com este mesmo estado? (s/n): ").lower()
            if continuar != 's':
                break
        
        # Perguntar se quer executar novamente com outro estado
        executar_novamente = input("\nDeseja executar com outro estado inicial? (s/n): ").lower()
        if executar_novamente != 's':
            print("Obrigado por usar o resolvedor de 8-puzzle!")
            break