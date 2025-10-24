import json
from pathlib import Path
import pandas as pd
import pickle
import importlib.util

BASE_DIR = Path(__file__).resolve().parent.parent

FAIRNESS_THRESHOLD: float | None = None
FAIRNESS_THRESHOLD_TYPE: str | None = None
MODEL: object | None = None
X_TEST: pd.DataFrame | None = None
Y_TEST: pd.DataFrame | None = None
FAIRNESS_MEASURE_FUNCTION: str | None = None
FAIRNESS_MEASURE: float | None = None


# Strategy "verify" in "fairClassification"
def verifying_acceptance_threshold() -> bool:
    if FAIRNESS_THRESHOLD_TYPE=="sup" :
        return FAIRNESS_MEASURE > FAIRNESS_THRESHOLD
    else :
        return FAIRNESS_MEASURE < FAIRNESS_THRESHOLD
    

# Evidence "level" in "fairClassification"
def threshold_level_is_defined () -> bool:
    global FAIRNESS_THRESHOLD
    global FAIRNESS_THRESHOLD_TYPE

    if (BASE_DIR / "requirements" / "fairness.json").exists():
        with open(BASE_DIR / "requirements" / "fairness.json", "r") as f:
            req = json.load(f)
            FAIRNESS_THRESHOLD = req['threshold']
            FAIRNESS_THRESHOLD_TYPE = req['threshold_type']
        return True    
    return False


# Strategy "fmetric" in "fairClassification"
def demographic_parity_measure() -> bool :
    global FAIRNESS_MEASURE
    
    # Ce bloc peut être décomposé en une évidence supplémentaire sous cette stratégie :
    # "Sensitive feature available" par exemple
    sensitive_test = pd.read_csv(BASE_DIR / "data" / "p_test.csv")

    y_pred = MODEL.predict(X_TEST)

    measure_path = Path(FAIRNESS_MEASURE_FUNCTION)
    spec = importlib.util.spec_from_file_location("fairness", measure_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    measurement = module.metric
    FAIRNESS_MEASURE = float(measurement(Y_TEST, y_pred, sensitive_test))
    
    return True


# Evidence "dataset" in "fairClassification"
def test_data_set_available() -> bool :
    global X_TEST, Y_TEST

    x_path = BASE_DIR / "data" / "X_test.csv"
    y_path = BASE_DIR / "data" / "y_test.csv"

    if x_path.exists() and y_path.exists():
        X_TEST = pd.read_csv(x_path)
        Y_TEST = pd.read_csv(y_path).values.ravel()
        return True
    return False


# Evidence "measurement" in "fairClassification"
def the_metric_measurement_available() -> bool :
    global FAIRNESS_MEASURE_FUNCTION
    
    if (BASE_DIR / "requirements" / "fairness.json").exists():
        with open(BASE_DIR / "requirements" / "fairness.json", "r") as f:
            req = json.load(f)
            FAIRNESS_MEASURE_FUNCTION = BASE_DIR / "metrics" / "fairness" / req["function"]
        return True
    return False


# Evidence "model" in "fairClassification"
def model_available() -> bool :
    global MODEL

    if (BASE_DIR / "model" / "model.pkl").exists():
        with open(BASE_DIR / "model" / "model.pkl", "rb") as f:
            MODEL = pickle.load(f)
        return True
    return False




# # Checker
# def checker(selected_experiment: str) -> str:
#     steps = [
#         ("Threshold level defined", threshold_level_is_defined),
#         ("Metric measurement available", metric_measurement_available),
#         ("Model available", lambda: model_available(selected_experiment)),
#         ("Test dataset available", lambda: test_data_set_available(selected_experiment)),
#         ("Fairness metric computed", lambda: demographic_parity_measure(selected_experiment)),
#         ("Threshold acceptance verified", verifying_acceptance_threshold),
#     ]
#     string = ""
#     all_passed = True
#     for name, step in steps:
#         try:
#             result = step()
#         except Exception as e:
#             string += f"[FAILED] {name} -> Exception: {e}</br>"
#             return string
#         if result:
#             string += f"[OK] {name}</br>"
#         else:
#             string += f"[FAILED] {name}</br>"
#             all_passed = False
#     if all_passed :
#         string += "[ALL PASSED]"
#     else :
#         string += "[FAILED]"
#     return string