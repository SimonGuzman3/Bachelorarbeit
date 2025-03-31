# Greedy algorithm (from NetworkX-bib)
import networkx as nx

#greedy-algorithm
def greedyRandom_coloring(G):
    return nx.coloring.greedy_color(G, strategy="random_sequential")
def run_greedy_multiple_times(G, times=1000):
    best_coloring = None
    min_colors = float('inf') # set min color to infinity
    
    for _ in range(times):
        # do coloring
        coloring = greedyRandom_coloring(G)
        
        # count used numbers
        num_colors = len(set(coloring.values()))
        
        # if colors used are less, then updated best color
        if num_colors < min_colors:
            min_colors = num_colors
            best_coloring = coloring
    
    # return best coloring
    return best_coloring

#greedy-algorithm LF
def greedyLF_coloring(G):
    return nx.coloring.greedy_color(G, strategy="largest_first")

#greedy-algorithm DSatur
def greedyDSatur_coloring(G):
    return nx.coloring.greedy_color(G, strategy="saturation_largest_first")

#Welsh Powell algorithm
def welsh_powell_coloring(G):
    # Sort nodes by their degree in descending order
    nodes_by_degree = sorted(G.nodes(), key=lambda node: len(list(G.neighbors(node))), reverse=True)
    coloring = {}
    
    # Assign a color to each node
    for node in nodes_by_degree:
        # find color of neighors
        neighbor_colors = {coloring.get(neighbor) for neighbor in G.neighbors(node) if neighbor in coloring}
        
        # assign lowest color not used by neighors
        color = 0
        while color in neighbor_colors:
            color += 1
        
        coloring[node] = color
    return coloring

