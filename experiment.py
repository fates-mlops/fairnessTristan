import pandas as pd
import json
import os
import importlib.util
from pathlib import Path
from sklearn.model_selection import train_test_split

####

# Récupérer les paramètres

with open("experiments/input.json", "r", encoding="utf-8") as f:
    input_param = json.load(f)

EXPERIMENT_NAME = input_param["experiment_name"]

# Data
DATA_SOURCE = input_param["data_source"]["dataset"]
DATA_PREPROCESSING = input_param["data_source"]["preprocessing"]
SPLIT_TARGET = input_param["split_param"]["target"] # Variable cible, qui constituera le Y
SPLIT_PROTECTED = input_param["split_param"]["protected"] # Variable protégée (fairness)
SPLIT_TEST_SIZE = input_param["split_param"]["test_size"] # Proportion de l'échantillon de test
SPLIT_SEED = input_param["split_param"]["seed"] # Graine aléatoire pour la réalisation du split

# Model
MODEL_TYPE = input_param["model"]["type"] # Référence à un fichier de modèle pour l'entrainement
MODEL_HYPERPARAMETERS = input_param["model"]["hyperparameters"] # Les hyperparamètres pour l'entrainement du modèle

# Metrics

PERFORMANCE_METRICS = input_param["performance_metrics"]
FAIRNESS_METRICS = input_param["fairness_metrics"]

# Corrections

FAIRNESS_TREATMENT = input_param["fairness_treatment"]

####

# Création du dossier de l'experiment.
experiment_path = f"experiments/{EXPERIMENT_NAME}"
#if not os.path.exists(experiment_path) :
os.makedirs(experiment_path)

# Lecture du jeu de données
dataset = pd.read_csv(f"data/{DATA_SOURCE}/data.csv")

# Prétraitement des données
preprocessing_path = Path(f"data/{DATA_SOURCE}/preprocessing/{DATA_PREPROCESSING}")
spec = importlib.util.spec_from_file_location("preprocessing", preprocessing_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
preprocess = module.preprocess
preprocessed_data = preprocess(dataset, experiment_path)

# Split des données
train, test = train_test_split(preprocessed_data, test_size=SPLIT_TEST_SIZE, random_state=SPLIT_SEED, shuffle=True)
X_train = train.drop(SPLIT_TARGET, axis=1)
y_train = train[SPLIT_TARGET]
X_test = test.drop(SPLIT_TARGET, axis=1)
y_test = test[SPLIT_TARGET]
p_train = train[SPLIT_PROTECTED] 
p_test = test[SPLIT_PROTECTED]

if "hide_protected" in FAIRNESS_TREATMENT :
    X_train = X_train.drop(SPLIT_PROTECTED, axis=1)
    X_test = X_test.drop(SPLIT_PROTECTED, axis=1)

os.makedirs(f"{experiment_path}/split")
X_train.to_csv(f"{experiment_path}/split/X_train.csv", index=None)
y_train.to_csv(f"{experiment_path}/split/y_train.csv", index=None)
X_test.to_csv(f"{experiment_path}/split/X_test.csv", index=None)
y_test.to_csv(f"{experiment_path}/split/y_test.csv", index=None)
p_train.to_csv(f"{experiment_path}/split/p_train.csv", index=None)
p_test.to_csv(f"{experiment_path}/split/p_test.csv", index=None)


# Entrainement du modèle
if MODEL_TYPE == "RandomForestClassifier":
    from sklearn.ensemble import RandomForestClassifier
    import pickle
    model = RandomForestClassifier(**MODEL_HYPERPARAMETERS)
    model.fit(X_train, y_train)
    with open(f"{experiment_path}/model.pkl", "wb") as f:
        pickle.dump(model, f)

# Prédictions
y_pred = model.predict(X_test)
pd.Series(y_pred).to_csv(f"{experiment_path}/y_pred.csv", index=None)

# Mesures
input_param["performance_measurments"] = {}
for metric in PERFORMANCE_METRICS :
    metric_path = Path(f"metrics/performance/{metric}")
    spec = importlib.util.spec_from_file_location("metric", metric_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metric_function = module.metric
    measure = metric_function(y_test, y_pred)
    input_param["performance_measurments"][metric[:-3]] = measure

input_param["fairness_measurments"] = {}
for metric in FAIRNESS_METRICS :
    metric_path = Path(f"metrics/fairness/{metric}")
    spec = importlib.util.spec_from_file_location("metric", metric_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metric_function = module.metric
    measure = metric_function(y_test, y_pred, p_test)
    input_param["fairness_measurments"][metric[:-3]] = measure



with open(f"{experiment_path}/parameters.json", "w", encoding="utf-8") as f:
    json.dump(input_param, f, indent=4, ensure_ascii=False)