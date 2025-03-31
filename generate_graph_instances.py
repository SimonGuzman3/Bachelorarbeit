import random, sys, os
import networkx as nx

def get_next_filename(graph_typ, directory):
    # search for existing files
    files = [f for f in os.listdir(directory) if f.startswith(f'{graph_typ}_') and f.endswith('.col')]
    # extract the numbers from filename and get highest
    numbers = []
    for file in files:
        try:
            number = int(file[len(graph_typ)+1:len(graph_typ)+5])  # filename format: results_0001.h5 -> 0002.h5 ...
            numbers.append(number)
        except ValueError:
            continue
    # if no file exists, start with 1
    if numbers:
        next_number = max(numbers) + 1
    else:
        next_number = 1
    return f'{graph_typ}_{next_number:04d}.col'

def save_graph(G, filename, directory):
    with open(os.path.join(directory, filename), 'w') as f:
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        f.write(f"p edge {num_nodes} {num_edges}\n")
        for u, v in G.edges():
            f.write(f"e {u+1} {v+1}\n")

# choose random graph-typ
def switch_case(x, n, p): # additional graphs: bipartier Graph, planarer Graph
    match x:
        case 1:
            G = nx.gnp_random_graph(n, p)  # Erdős-Rényi Random Graph
            return G, "random"
        case 2:
            m = random.randint(5, 100) # random numbers of neighbors
            G = nx.barabasi_albert_graph(n, m)  # Barabási-Albert Scale-Free Graph
            return G, "barabasi"
        case 3:
            k = random.randint(5,50) # random number of neighbors
            G = nx.watts_strogatz_graph(n, k, p)  # Watts-Strogatz Small-World
            return G, "watts_strogatz"
        case _:
            print("No matching graph type")
            sys.exit()

times = 1 # how many graphs are generated

save_directory = 'new_graph_instances'
# Debug-Check, ob die Datei existiert
if os.path.exists(save_directory):
    print(f"Datei erfolgreich gespeichert: {save_directory}")
else:
    print(f"Fehler: Datei nicht gespeichert! {save_directory}")
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

for i in range(times):
    random_graph = random.randint(1, 3) 
    n = random.randint(500, 10000) # number nodes
    if random_graph == 1:
        p = random.uniform(0, 0.3) # probability of edge between two nodes
    else:
        p = random.uniform(0, 1)  # probability for edges
    G, graph_typ = switch_case(random_graph, n, p) # create graph with switch_case
    filename = get_next_filename(graph_typ, save_directory)

    # create thousends seperator
    formatted_nodes = f"{n:,}".replace(",", ".")
    formatted_edges = f"{G.number_of_edges():,}".replace(",", ".")

    # Print nodes and edges with thousands separator
    print(f"Graph {i+1}: n = {formatted_nodes}, e = {formatted_edges}")

    # save graphs as .col with DIMACS standart
    save_graph(G, filename, save_directory)
