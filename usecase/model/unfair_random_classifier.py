import pickle
import random
import pandas as pd
from pathlib import Path
import json
import importlib.util

BASE_DIR = Path(__file__).resolve().parent.parent

class UnfairRandomClassifier:
    def __init__(self, classes, protected):
        self.classes = classes
        self.protected = protected

    def predict(self, X):
        return [random.choice(self.classes) if X[self.protected].iloc[elem] else False for elem in range(len(X))]

model = UnfairRandomClassifier(classes=[False, True], protected="sex")
with open(BASE_DIR / "model" / "model.pkl", "wb") as f:
    pickle.dump(model, f)

with open(BASE_DIR / "model" / "model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

X_test = pd.read_csv(BASE_DIR / "data" / "X_test.csv")

y_pred = loaded_model.predict(X_test)

y_pred = pd.Series(y_pred)

y_pred.to_csv(BASE_DIR / "data" / "y_pred.csv", index=False)

with open(BASE_DIR / "model" / "parameters.json", "r", encoding="utf-8") as f:
    param = json.load(f)

y_test = pd.read_csv(BASE_DIR / "data" / "y_test.csv")

for metric in param["performance_metrics"] :
    metric_path = BASE_DIR / "metrics" / "performance" / f"{metric}.py"
    spec = importlib.util.spec_from_file_location("metric", metric_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metric_function = module.metric
    measure = metric_function(y_test, y_pred)
    param["performance_measurments"][metric] = measure

p_test = pd.read_csv(BASE_DIR / "data" / "p_test.csv")

for metric in param["fairness_metrics"] :
    metric_path = BASE_DIR / "metrics" / "fairness" / f"{metric}.py"
    spec = importlib.util.spec_from_file_location("metric", metric_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metric_function = module.metric
    measure = metric_function(y_test, y_pred, p_test)
    param["fairness_measurments"][metric] = measure

with open(BASE_DIR / "model" / "parameters.json", "w", encoding="utf-8") as f:
    json.dump(param, f, indent=4, ensure_ascii=False)