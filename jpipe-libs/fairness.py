import json
from pathlib import Path
import pandas as pd
import pickle
import importlib.util

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

    if Path("../3_requirements/fairness.json").exists():
        with open("../3_requirements/fairness.json", "r") as f:
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
    with open("../3_requirements/fairness.json", "r") as f:
        req = json.load(f)
    sensitive_feature_name = req["sensitive_feature"]
    sensitive_test = X_TEST[sensitive_feature_name]

    y_pred = MODEL.predict(X_TEST)

    measure_path = Path(FAIRNESS_MEASURE_FUNCTION)
    spec = importlib.util.spec_from_file_location("fairness", measure_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    measurement = module.measurement

    FAIRNESS_MEASURE = float(measurement(Y_TEST, y_pred, sensitive_test))
    
    return True

# Evidence "dataset" in "fairClassification"
def test_data_set_available() -> bool :
    global X_TEST, Y_TEST

    x_path = Path("../1_data/3_split/X_test.csv")
    y_path = Path("../1_data/3_split/y_test.csv")

    if x_path.exists() and y_path.exists():
        X_TEST = pd.read_csv(x_path, index_col=0)
        Y_TEST = pd.read_csv(y_path, index_col=0).values.ravel()
        return True
    return False

# Evidence "measurement" in "fairClassification"
def metric_measurement_available() -> bool :
    global FAIRNESS_MEASURE_FUNCTION

    if Path("../3_requirements/fairness.json").exists():
        with open("../3_requirements/fairness.json", "r") as f:
            req = json.load(f)
            FAIRNESS_MEASURE_FUNCTION = f"../3_requirements/{req['function']}"
        return True
    return False



# Evidence "model" in "fairClassification"
def model_available() -> bool :
    global MODEL

    if Path("../2_models/random_forest_model.pkl").exists():
        with open("../2_models/random_forest_model.pkl", "rb") as f:
            MODEL = pickle.load(f)
        return True
    return False