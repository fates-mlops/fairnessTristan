import json
from pathlib import Path
import pandas as pd
import pickle
import importlib.util

SELECTED_EXPERIMENT = "RF_Income_Fair"

FAIRNESS_THRESHOLD: float | None = None
FAIRNESS_THRESHOLD_TYPE: str | None = None


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

    if Path("../requirements/fairness.json").exists():
        with open("../requirements/fairness.json", "r") as f:
            req = json.load(f)
            FAIRNESS_THRESHOLD = req['threshold']
            FAIRNESS_THRESHOLD_TYPE = req['threshold_type']
        return True    
    return False

MODEL: object | None = None

X_TEST: pd.DataFrame | None = None
Y_TEST: pd.DataFrame | None = None

FAIRNESS_MEASURE_FUNCTION: str | None = None

FAIRNESS_MEASURE: float | None = None

# Strategy "fmetric" in "fairClassification"
def demographic_parity_measure() -> bool :
    global FAIRNESS_MEASURE
    
    # Ce bloc peut être décomposé en une évidence supplémentaire sous cette stratégie :
    # "Sensitive feature available" par exemple
    sensitive_test = pd.read_csv(f"../experiments/{SELECTED_EXPERIMENT}/split/p_test.csv")

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

    x_path = Path(f"../experiments/{SELECTED_EXPERIMENT}/split/X_test.csv")
    y_path = Path(f"../experiments/{SELECTED_EXPERIMENT}/split/y_test.csv")

    if x_path.exists() and y_path.exists():
        X_TEST = pd.read_csv(x_path)
        Y_TEST = pd.read_csv(y_path).values.ravel()
        return True
    return False

# Evidence "measurement" in "fairClassification"
def metric_measurement_available() -> bool :
    global FAIRNESS_MEASURE_FUNCTION

    if Path("../requirements/fairness.json").exists():
        with open("../requirements/fairness.json", "r") as f:
            req = json.load(f)
            FAIRNESS_MEASURE_FUNCTION = f"../metrics/fairness/{req['function']}"
        return True
    return False



# Evidence "model" in "fairClassification"
def model_available() -> bool :
    global MODEL

    if Path(f"../experiments/{SELECTED_EXPERIMENT}/model.pkl").exists():
        with open(f"../experiments/{SELECTED_EXPERIMENT}/model.pkl", "rb") as f:
            MODEL = pickle.load(f)
        return True
    return False