import pandas as pd

def prepare_ml_data(all_results):
    data = []
    alpha = 3 # set importance of algorithm speed

    for index, entry in enumerate(all_results):
        graph_id = f"graph_{index + 1}"
        
        graph_info = entry['graph_info']  # Access to the inner dict with graph information
        result_data = entry['result']     # access to the algorithm result
        
        features = {
            'number_edges': graph_info['number_edges'],
            'density': graph_info['density'],
            'min_degree': graph_info['min_degree'],
            'degree_ratio': graph_info['degree_ratio'],
            'clustering_coeff': graph_info['clustering_coeff'],
            'graph_typ': graph_info['graph_typ'],
        }

        # Add algorithmic results
        for algorithm, result in result_data.items():
            if f'{algorithm}_time' in ['WP_time']:
                features[f'{algorithm}_time'] = result['time']

        # Find the best algorithm based on number of colors and runtime
        best_algorithm = min(
            result_data,
            key=lambda alg: (
                result_data[alg]['colors'] + alpha * (result_data[alg]['time']) # apha = 3 -> 1 second delay is as important as 3 additional colors
            )
        )
        
        features['best_algorithm'] = best_algorithm
        data.append(features)

    return pd.DataFrame(data)



