from collections import deque  # Importa deque para fila eficiente
import time  # Para medir tempo de execução

# Estado objetivo do 8-puzzle
GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)

def show(state):
    """Exibe o estado atual do puzzle em formato 3x3"""
    for i in range(0, 9, 3):  # Percorre linhas
        print(state[i], state[i+1], state[i+2])  # Imprime cada linha
    print()  # Espaço entre estados

def find_zero(state):
    """Retorna a posição do espaço vazio (0) no estado atual"""
    return state.index(0)  # Usa método de lista para encontrar índice

def get_moves(z_pos):
    """Gera movimentos válidos com base na posição atual do zero"""
    moves = []
    row, col = divmod(z_pos, 3)  # Converte posição linear para (linha, coluna)
    
    # Verifica movimentos possíveis com base nas bordas
    if row > 0: moves.append('up')     # Pode mover para cima se não estiver na linha superior
    if row < 2: moves.append('down')   # Pode mover para baixo se não estiver na linha inferior
    if col > 0: moves.append('left')   # Pode mover para esquerda se não estiver na coluna esquerda
    if col < 2: moves.append('right')  # Pode mover para direita se não estiver na coluna direita
    return moves

def apply_move(state, move, z_pos):
    """Aplica um movimento ao estado atual, retornando novo estado"""
    # Calcula nova posição do zero baseado no movimento
    if move == 'up': new_pos = z_pos - 3
    elif move == 'down': new_pos = z_pos + 3
    elif move == 'left': new_pos = z_pos - 1
    elif move == 'right': new_pos = z_pos + 1
    else: return None  # Movimento inválido
    
    # Cria novo estado trocando a posição do zero
    state_list = list(state)  # Converte tupla em lista para modificação
    state_list[z_pos], state_list[new_pos] = state_list[new_pos], state_list[z_pos]  # Troca posições
    return tuple(state_list)  # Retorna como tupla (imutável)

def is_solvable(state):
    """Verifica se o puzzle é solucionável contando inversões"""
    seq = [x for x in state if x != 0]  # Remove o zero da sequência
    inversions = sum(
        1 for i in range(len(seq)) 
        for j in range(i+1, len(seq)) 
        if seq[i] > seq[j]  # Conta pares invertidos
    )
    return inversions % 2 == 0  # Solucionável se número de inversões for par

def bfs_metrics(start):
    """Implementação do BFS com coleta de métricas"""
    
    # Verifica se o estado é solucionável
    if not is_solvable(start):
        print("Esse estado inicial é insolúvel!")
        return None, None
    
    # Caso o estado inicial já seja o objetivo
    if start == GOAL:
        return [start], {"nodes_expanded": 0, "frontier_max": 1, "solution_len": 0}
    
    # Inicializa estruturas de dados para busca
    queue = deque([[start]])  # Fila de caminhos a explorar (inicia com estado inicial)
    visited = {start}         # Conjunto de estados visitados
    
    # Variáveis para métricas
    nodes_expanded = 0    # Contador de nós expandidos
    frontier_max = 1      # Tamanho máximo da fronteira
    start_time = time.time()  # Marca tempo inicial
    
    while queue:  # Enquanto houver estados para explorar
        path = queue.popleft()  # Remove o primeiro caminho da fila (FIFO)
        current = path[-1]      # Pega o estado atual do caminho
        nodes_expanded += 1     # Incrementa contador de expansões
        
        z_pos = find_zero(current)  # Encontra posição do zero
        
        # Gera e explora todos os movimentos possíveis
        for move in get_moves(z_pos):
            new_state = apply_move(current, move, z_pos)  # Gera novo estado
            
            # Ignora estados inválidos ou já visitados
            if new_state is None or new_state in visited:
                continue
            
            # Verifica se encontrou solução
            if new_state == GOAL:
                solution = path + [new_state]  # Cria caminho completo
                metrics = {
                    "nodes_expanded": nodes_expanded,
                    "frontier_max": max(frontier_max, len(queue)),
                    "solution_len": len(solution) - 1,  # Número de movimentos = passos - 1
                    "time": time.time() - start_time
                }
                return solution, metrics  # Retorna solução e métricas
            
            # Atualiza estruturas para continuar busca
            visited.add(new_state)  # Marca novo estado como visitado
            queue.append(path + [new_state])  # Adiciona novo caminho à fila
            frontier_max = max(frontier_max, len(queue))  # Atualiza tamanho máximo da fronteira
    
    # Caso não encontre solução (teoricamente não ocorre para estados solucionáveis)
    return None, {"nodes_expanded": nodes_expanded, "frontier_max": frontier_max, "solution_len": None}

# Teste com estado inicial específico
start_state = (1, 2, 3,
               6, 4, 0,
               7, 8, 5)

print("Estado inicial:")
show(start_state)

# Executa busca
solution, metrics = bfs_metrics(start_state)

# Exibe resultados
if solution:
    print(f"Solução encontrada em {metrics['solution_len']} movimentos:")
    for step, state in enumerate(solution):
        print(f"Passo {step}:")
        show(state)
    print("Métricas:", metrics)
else:
    print("Nenhuma solução encontrada!")