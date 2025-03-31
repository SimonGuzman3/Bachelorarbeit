from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from graph_result_processor import prepare_ml_data
from algorithmComparision import get_data
import h5py, os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# function for loading results grom hdF5 file
def load_results_hdf5(directory):
    all_results = []
    # go through all files in directory
    for filename in os.listdir(directory):
        if filename.endswith('.h5'):
            file_path = os.path.join(directory, filename)
            with h5py.File(file_path, 'r') as f:
                for graph_key in f.keys():
                    graph_group = f[graph_key]
                    graph_info = {
                        'number_edges': graph_group['number_edges'][()],
                        'density': graph_group['density'][()],
                        'min_degree': graph_group['min_degree'][()],
                        'degree_ratio': graph_group['degree_ratio'][()],
                        'clustering_coeff': graph_group['clustering_coeff'][()],
                        'graph_typ': graph_group['graph_typ'][()],
                    }
                    # recreate graph from edgelist saved in graph_group['graph'], needed for graph plotting
                    # G = nx.Graph()
                    # G.add_edges_from(graph_group['graph'][:])
                    # graph_info['graph'] = G

                    result_data = {}
                    algorithms_group = graph_group['algorithms']
                    for algo_key in algorithms_group.keys():
                        result_data[algo_key] = {
                            'time': algorithms_group[algo_key]['time'][()],
                            'colors': algorithms_group[algo_key]['colors'][()],
                            'correct': algorithms_group[algo_key]['correct'][()]
                        }
                    all_results.append({
                        'graph_info': graph_info,
                        'result': result_data
                    })
    return all_results

# Hyperparameter-tuning-function random forest
def rf_hyperparameter_tuning(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 3, 4],
        'max_features': ['sqrt', 'log2']
    }
    # RandomizedSearchCV initialize
    randomized_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=param_grid,
        cv=5,  # 5-times Cross-Validation
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )
    randomized_search.fit(X_train, y_train)
    return randomized_search.best_estimator_

# Hyperparameter-tuning-function for GradianBoosting
def gbg_hyperparameter_tuning(X_train, y_train):
    # Hyperparameter-Raum definieren
    param_grid = {
        'n_estimators': [50, 100, 150, 200, 300],
        'max_depth': [5, 7, 10, 15],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'min_samples_split': [10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2'],
        'subsample': [0.4, 0.6, 0.8, 1.0]
    }

    # Randomized Search aufrufen
    randomized_search = RandomizedSearchCV(
        estimator=GradientBoostingClassifier(random_state=42), 
        param_distributions=param_grid,
        cv=5,  # 5-fache Cross-Validation
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )
    randomized_search.fit(X_train, y_train)
    return randomized_search.best_estimator_

def mlp_hyperparameter_tuning(X_train, y_train):
    # Define hyperparameter grid
    param_grid = {
        'hidden_layer_sizes': [(256, 128, 64), (128, 64, 32)],
        'activation': ['relu'],
        'alpha': [0.001, 0.01, 0.1],
        'learning_rate_init': [0.0005, 0.0001],
        'max_iter': [1000, 1500],
    }

    # GridSearchCV for hyperparameter tuning
    randomized_search = RandomizedSearchCV(
        estimator=MLPClassifier(random_state=42, verbose=True),
        param_distributions=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    randomized_search.fit(X_train, y_train)
    return randomized_search.best_estimator_

# Cross-Validation-Function
def evaluate_with_cross_validation(model, X, y):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    return scores

# Random Forest training
def rf_training(X, y):
    # Convert categories to numbers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/Test-Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    # RandomForest-Training
    best_rf_model = rf_hyperparameter_tuning(X_train, y_train)

    # Cross-validation with the optimized model
    cross_val_scores = evaluate_with_cross_validation(best_rf_model, X, y)

    # prediction and evaluation
    # time_greedyLF, greedyLF_color = time_measure(greedyLF_coloring, G)
    y_pred_rf = best_rf_model.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)

    return accuracy_rf, y_pred_rf

# XGBoost training
def gbg_training(X, y):
    # Convert categories to numbers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/Test-Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    # RandomForest-Training
    best_gbg_model = gbg_hyperparameter_tuning(X_train, y_train)

    # Cross-validation with the optimized model
    cross_val_scores = evaluate_with_cross_validation(best_gbg_model, X, y)

    # prediction and evaluation
    y_pred_gbg = best_gbg_model.predict(X_test)
    accuracy_gbg = accuracy_score(y_test, y_pred_gbg)
    
    return accuracy_gbg, y_pred_gbg

def mlp_training(X, y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/Test-Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    # Create MLP model
    mlp_model = MLPClassifier(hidden_layer_sizes=(256, 128, 64), 
                               activation='relu', 
                               solver='adam', 
                               max_iter=500, 
                               random_state=42)

    # Train the model
    # mlp_model.fit(X_train, y_train)
    best_model = mlp_hyperparameter_tuning(X_train, y_train)

    # Cross-validation
    scores = cross_val_score(mlp_model, X_scaled, y_encoded, cv=10, scoring='accuracy')

    # Test set evaluation
    y_pred = best_model.predict(X_test)
    # y_pred = mlp_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy, y_pred

def save_dataframe_to_txt(df, directory, filename="ai_data.txt"):

    if not os.path.exists(directory):
        os.makedirs(directory)  # Ordner erstellen, falls nicht vorhanden
    
    file_path = os.path.join(directory, filename)
    
    df.to_csv(file_path, sep='\t', index=False)  # Speichert als Tab-getrennte Datei

# get_data() # coloring all graphs ------ execute only once ------
directory = 'graph_tables'  # the folder with HDF5-files
loaded_results = load_results_hdf5(directory)

# prepare data
df = prepare_ml_data(loaded_results)

# Count the frequency of the best algorithms
best_algorithms = df['best_algorithm']
algorithm_counts = Counter(best_algorithms)


# extract features and labels
X = df.drop(columns=['best_algorithm','graph_typ'])
y = df['best_algorithm']
save_dataframe_to_txt(df, "dataframe")

# print accuracy of ai models
nn_accuracy, mlp_preds = mlp_training(X, y)
rf_accuracy, rf_preds = rf_training(X, y)
gbg_accuracy, gbg_preds = gbg_training(X, y)
print(f'Accuracy NN: {nn_accuracy:.3f}') # print accuracy of NN
print(f'Accuracy RF: {rf_accuracy:.3f}') # print accuracy of random forest
print(f'Accuracy GBG: {gbg_accuracy:.3f}') # print accuracy of XGBoost

# Find the model with the highest accuracy
model_accuracies = {
    "NN": nn_accuracy,
    "RF": rf_accuracy,
    "GBG": gbg_accuracy,
}
best_model_name = max(model_accuracies, key=model_accuracies.get)
best_accuracy = model_accuracies[best_model_name]

# print winner of AI-Model training and accuracy
print(f'Best AI-Model: {best_model_name} with an accuracy of: {best_accuracy:.3f}')

print(algorithm_counts)


# print grafic for algorithm quality comparision

# We remember the original indices of the test data
_, X_test_indices = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
# convert algorithms into numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Unified mapping for algorithms
algorithms = {
    'WP': 'WP',
    'LF': 'GreedyLF',
    'DSatur': 'GreedyDSatur',
    'GreedyMult': 'GreedyRMulti'
}

# Initialize points
all_points = {
    'heusterics': {name: {'time': 0.0, 'colors': 0} for name in algorithms.keys()},
    'rf': {'time': 0.0, 'colors': 0},
    'gbg': {'time': 0.0, 'colors': 0},
    'mlp': {'time': 0.0, 'colors': 0},
}

for i, rf_pred, gbg_pred, mlp_pred in zip(X_test_indices, rf_preds, gbg_preds, mlp_preds):
    result = loaded_results[i]

    # Values ​​per heuristic
    for short, full in algorithms.items():
        alg_result = result['result'].get(full)
        if alg_result:
            all_points['heusterics'][short]['time'] += alg_result['time']
            all_points['heusterics'][short]['colors'] += alg_result['colors']

    # Values ​​per AI predictions
    for model_name, pred in zip(['rf', 'gbg', 'mlp'], [rf_pred, gbg_pred, mlp_pred]):
        pred_full = label_encoder.inverse_transform([pred])[0]
        pred_short = {v: k for k, v in algorithms.items()}.get(pred_full)
        if pred_short and pred_full in result['result']:
            alg_result = result['result'][pred_full]
            all_points[model_name]['time'] += alg_result['time']
            all_points[model_name]['colors'] += alg_result['colors']

# plotting
plt.figure(figsize=(10, 6))

# Color palette
custom_colors = {
    'WP': '#1f77b4',         # blue
    'LF': '#ff7f0e',         # orange
    'DSatur': '#2ca02c',     # green
    'GreedyMult': '#d62728', # red

    'RF': '#000000',         # black
    'GBG': '#17becf',        # turquoise
    'MLP': '#e377c2'         # pink
}

# AI points
for model_name in ['rf', 'gbg', 'mlp']:
    label = model_name.upper()
    data = all_points[model_name]
    x, y = data['time'], data['colors']
    plt.scatter(x, y, label=label, marker='X', s=50,
                color=custom_colors[label], linewidth=1.2)

# Heuristic points
for alg, data in all_points['heusterics'].items():
    x, y = data['time'], data['colors']
    plt.scatter(x, y, label=alg,
                color=custom_colors[alg], s=60)

plt.xscale('log')
plt.xlabel("Gesamte Zeit (Sekunden) [log]")
plt.ylabel("Summe der verwendeten Farben")
plt.title("Heuristiken und KI-Vorhersagen: Zeit vs. Farben")
plt.grid(True, which="major", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.legend()
plt.show()

