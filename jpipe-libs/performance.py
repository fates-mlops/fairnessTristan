import json
from pathlib import Path
import pandas as pd
import pickle
import importlib.util

SELECTED_EXPERIMENT = "RF_Income_Fair"

PERFORMANCE_THRESHOLD: float | None = None
PERFORMANCE_THRESHOLD_TYPE: str | None = None


# Strategy "verify" in "performance"
def verifying_acceptance_threshold() -> bool:
    if PERFORMANCE_THRESHOLD_TYPE=="sup" :
        return PERFORMANCE_MEASURE > PERFORMANCE_THRESHOLD
    else :
        return PERFORMANCE_MEASURE < PERFORMANCE_THRESHOLD
    
# Evidence "level" in "performance"
def threshold_level_is_defined() -> bool:
    global PERFORMANCE_THRESHOLD
    global PERFORMANCE_THRESHOLD_TYPE

    if Path("../requirements/performance.json").exists():
        with open("../requirements/performance.json", "r") as f:
            req = json.load(f)
            PERFORMANCE_THRESHOLD = req['threshold']
            PERFORMANCE_THRESHOLD_TYPE = req['threshold_type']
        return True    
    return False

MODEL: object | None = None

X_TEST: pd.DataFrame | None = None
Y_TEST: pd.DataFrame | None = None

PERFORMANCE_MEASURE_FUNCTION: str | None = None

PERFORMANCE_MEASURE: float | None = None

# Strategy "pmetric" in "performance"
def f1_score_measure() -> bool :
    global PERFORMANCE_MEASURE

    y_pred = MODEL.predict(X_TEST)

    measure_path = Path(PERFORMANCE_MEASURE_FUNCTION)
    spec = importlib.util.spec_from_file_location("performance", measure_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    measurement = module.metric
    PERFORMANCE_MEASURE = float(measurement(Y_TEST, y_pred))
    
    return True

# Evidence "dataset" in "performance"
def test_data_set_available() -> bool :
    global X_TEST, Y_TEST

    x_path = Path(f"../experiments/{SELECTED_EXPERIMENT}/split/X_test.csv")
    y_path = Path(f"../experiments/{SELECTED_EXPERIMENT}/split/y_test.csv")

    if x_path.exists() and y_path.exists():
        X_TEST = pd.read_csv(x_path)
        Y_TEST = pd.read_csv(y_path).values.ravel()
        return True
    return False

# Evidence "measurement" in "performance"
def metric_measurement_available() -> bool :
    global PERFORMANCE_MEASURE_FUNCTION

    if Path("../requirements/performance.json").exists():
        with open("../requirements/performance.json", "r") as f:
            req = json.load(f)
            PERFORMANCE_MEASURE_FUNCTION = f"../metrics/performance/{req['function']}"
        return True
    return False



# Evidence "model" in "performance"
def model_available() -> bool :
    global MODEL

    if Path(f"../experiments/{SELECTED_EXPERIMENT}/model.pkl").exists():
        with open(f"../experiments/{SELECTED_EXPERIMENT}/model.pkl", "rb") as f:
            MODEL = pickle.load(f)
        return True
    return False