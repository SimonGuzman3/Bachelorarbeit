import time, h5py, os
import networkx as nx
from coloring_algorithms import run_greedy_multiple_times, greedyLF_coloring, greedyDSatur_coloring, welsh_powell_coloring # import algorithms 


# funktion for determining the runtime
def time_measure(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    runtime = end_time - start_time
    return runtime, result

# funktion to check correct coloring
def is_coloring_correct(G, colors):
    # check if adjacent nodes are different colors
    for u, v in G.edges():
        if colors[u] == colors[v]:  # if adjacent nodes are same color
            return False
    return True

# find highest file number
def get_next_filename(directory):
    # search for existing files
    files = [f for f in os.listdir(directory) if f.startswith('results_') and f.endswith('.h5')]
    # extract the numbers from filename and get highest
    numbers = []
    for file in files:
        try:
            number = int(file[8:12])  # filename format: results_0001.h5 -> 0002.h5 ...
            numbers.append(number)
        except ValueError:
            continue
    # if no file exists, start with 1
    if numbers:
        next_number = max(numbers) + 1
    else:
        next_number = 1
    return f'results_{next_number:04d}.h5'

# get results
def get_data():
    directory = 'graph_tables'
    filename = 'graph_data.h5'
    all_results = {}
    # other test instances
    graph_file = 'graph_testinstances'

    # processing graph instances 
    for i, graph_file_name in enumerate(os.listdir(graph_file)):
        if graph_file_name.endswith('.col') or graph_file_name.endswith('.txt'):
            filepath = os.path.join(graph_file, graph_file_name)
            G = load_graph_file(filepath)
            graph_typ = os.path.splitext(os.path.basename(graph_file_name))[0]  # file name without file type (.col/.txt)
            process_graph(G, graph_typ, i, all_results)
            print(f'Graph {i} completed')

    # save in h5 file
    save_results_to_hdf5(directory, filename, all_results)

# safe data as hdf5 
def save_results_to_hdf5(directory, filename, all_results):
    with h5py.File(os.path.join(directory, filename), 'w') as f:
        for graph_id, graph_data in all_results.items():
            group = f.create_group(graph_id)
            group.create_dataset('graph', data=graph_data['graph'])
            group.create_dataset('graph_typ', data=graph_data['graph_typ'])
            group.create_dataset('n', data=graph_data['n'])
            group.create_dataset('number_edges', data=graph_data['number_edges'])
            group.create_dataset('density', data=graph_data['density'])
            group.create_dataset('max_degree', data=graph_data['max_degree'])
            group.create_dataset('min_degree', data=graph_data['min_degree'])
            group.create_dataset('avg_degree', data=graph_data['avg_degree'])
            group.create_dataset('degree_ratio', data=graph_data['degree_ratio'])
            group.create_dataset('clustering_coeff', data=graph_data['clustering_coeff'])
            
            algorithms_group = group.create_group('algorithms')
            for algorithm, result_data in graph_data['algorithms'].items():
                alg_group = algorithms_group.create_group(algorithm)
                alg_group.create_dataset('time', data=result_data['time'])
                alg_group.create_dataset('colors', data=result_data['colors'])
                alg_group.create_dataset('correct', data=result_data['correct'])

# Processes a graph, executes the algorithms and stores the results 
def process_graph(G, graph_typ, index, all_results):
    # graph characteristics
    number_nodes = G.number_of_nodes()
    number_edges = G.number_of_edges()
    edges = list(G.edges())
    density = nx.density(G)
    max_degree = max(dict(G.degree()).values())
    min_degree = min(dict(G.degree()).values())
    avg_degree = sum(dict(G.degree()).values()) / max(number_nodes,1)
    degree_ratio = max_degree / (avg_degree + 1e-5)  # Avoiding division by zero
    clustering_coeff = nx.average_clustering(G)

    
    # algorithm execution and time measurement
    time_greedyLF, greedyLF_color = time_measure(greedyLF_coloring, G)
    time_wp, wp_color = time_measure(welsh_powell_coloring, G)
    time_greedyR, greedyR_color = time_measure(run_greedy_multiple_times, G)
    time_greedyDSatur, greedyDSatur_color = time_measure(greedyDSatur_coloring, G)

    result = {}
    result['GreedyRMulti'] = {
        'time': time_greedyR,
        'colors': max(greedyR_color.values()) + 1,
        'correct': is_coloring_correct(G, greedyR_color)
    }

    result['GreedyDSatur'] = {
        'time': time_greedyDSatur,
        'colors': max(greedyDSatur_color.values()) + 1,
        'correct': is_coloring_correct(G, greedyDSatur_color)
    }

    result['GreedyLF'] = {
        'time': time_greedyLF,
        'colors': max(greedyLF_color.values()) + 1,
        'correct': is_coloring_correct(G, greedyLF_color)
    }

    result['WP'] = {
        'time': time_wp,
        'colors': max(wp_color.values()) + 1,
        'correct': is_coloring_correct(G, wp_color)
    }

    # safe results as multi-array
    all_results[f'graph_{index + 1}'] = {
        'graph': edges,
        'graph_typ': graph_typ,
        'n': number_nodes,
        'number_edges': number_edges,
        'density': density,
        'max_degree': max_degree,
        'min_degree': min_degree,
        'avg_degree': avg_degree,
        'degree_ratio': degree_ratio,
        'clustering_coeff': clustering_coeff,
        'algorithms': result
    }

# load files
def load_graph_file(filepath):
    G = nx.Graph()
    if filepath.endswith('.col'):  # Handle .col files
        with open(filepath, 'r') as file:
            for line in file:
                if line.startswith('c'):
                    continue  # skip comments
                elif line.startswith('p'):
                    _, _, num_nodes, _ = line.split()
                    num_nodes = int(num_nodes)
                    G.add_nodes_from(range(1, num_nodes + 1))
                elif line.startswith('e'):
                    _, u, v = line.split()
                    G.add_edge(int(u), int(v))

    if filepath.endswith('.txt'):  # Handle .txt files
        with open(filepath, 'r') as file:
            for line in file:
                if line.startswith('c'):
                    continue  # skip comments
                elif line.startswith('p'):
                    _, _, num_nodes, _ = line.split()
                    num_nodes = int(num_nodes)
                    G.add_nodes_from(range(1, num_nodes + 1))
                elif line.startswith('e'):
                    _, u, v = line.split()
                    G.add_edge(int(u), int(v))

    return G
