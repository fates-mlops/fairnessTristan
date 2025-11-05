from sklearn.dummy import DummyClassifier
import pickle
import pandas as pd
from pathlib import Path
import json
import importlib.util

BASE_DIR = Path(__file__).resolve().parent.parent

#Récup des données d'entrainement
X_train = pd.read_csv(BASE_DIR / "data" / "X_train.csv")
y_train = pd.read_csv(BASE_DIR / "data" / "y_train.csv")
# Entrainement du modele
model = DummyClassifier(strategy="uniform")
model.fit(X_train, y_train)
# Sauvegarde du modèle
with open(BASE_DIR / "model" / "model.pkl", "wb") as f:
    pickle.dump(model, f)
# Rechargement du modèle
with open(BASE_DIR / "model" / "model.pkl", "rb") as f:
    loaded_model = pickle.load(f)
# Récup des input pour les prédictions
X_test = pd.read_csv(BASE_DIR / "data" / "X_test.csv")
# Prédictions du modèle
y_pred = loaded_model.predict(X_test)
y_pred = pd.Series(y_pred)
y_pred.to_csv(BASE_DIR / "data" / "y_pred.csv", index=False)

# Récup des données de test
y_test = pd.read_csv(BASE_DIR / "data" / "y_test.csv")
# Récup des métriques
with open(BASE_DIR / "model" / "parameters.json", "r", encoding="utf-8") as f:
    param = json.load(f)
# Calcul des métriques de perf
for metric in param["performance_metrics"] :
    metric_path = BASE_DIR / "metrics" / "performance" / f"{metric}.py"
    spec = importlib.util.spec_from_file_location("metric", metric_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metric_function = module.metric
    measure = metric_function(y_test, y_pred)
    param["performance_measurments"][metric] = measure
# Récup de la classe protégée
p_test = pd.read_csv(BASE_DIR / "data" / "p_test.csv")
# Calcul des métriques de fairness sur cette classe protégée
for metric in param["fairness_metrics"] :
    metric_path = BASE_DIR / "metrics" / "fairness" / f"{metric}.py"
    spec = importlib.util.spec_from_file_location("metric", metric_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metric_function = module.metric
    measure = metric_function(y_test, y_pred, p_test)
    param["fairness_measurments"][metric] = measure
# Sauvegarde des nouvelles caractéristiques du modèle
with open(BASE_DIR / "model" / "parameters.json", "w", encoding="utf-8") as f:
    json.dump(param, f, indent=4, ensure_ascii=False)