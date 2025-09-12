import json
from pathlib import Path
import pandas as pd
import pickle
import importlib.util

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

PERFORMANCE_THRESHOLD: float | None = None
PERFORMANCE_THRESHOLD_TYPE: str | None = None
MODEL: object | None = None
X_TEST: pd.DataFrame | None = None
Y_TEST: pd.DataFrame | None = None
PERFORMANCE_MEASURE_FUNCTION: str | None = None
PERFORMANCE_MEASURE: float | None = None


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

    if (PROJECT_ROOT / "requirements" / "performance.json").exists():
        with open(PROJECT_ROOT / "requirements" / "performance.json", "r") as f:
            req = json.load(f)
            PERFORMANCE_THRESHOLD = req['threshold']
            PERFORMANCE_THRESHOLD_TYPE = req['threshold_type']
        return True    
    return False


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
def test_data_set_available(selected_experiment: str) -> bool :
    global X_TEST, Y_TEST

    x_path = PROJECT_ROOT / "experiments" / selected_experiment / "split" / "X_test.csv"
    y_path = PROJECT_ROOT / "experiments" / selected_experiment / "split" / "y_test.csv"

    if x_path.exists() and y_path.exists():
        X_TEST = pd.read_csv(x_path)
        Y_TEST = pd.read_csv(y_path).values.ravel()
        return True
    return False


# Evidence "measurement" in "performance"
def metric_measurement_available() -> bool :
    global PERFORMANCE_MEASURE_FUNCTION

    if (PROJECT_ROOT / "requirements" / "performance.json").exists():
        with open(PROJECT_ROOT / "requirements" / "performance.json", "r") as f:
            req = json.load(f)
            PERFORMANCE_MEASURE_FUNCTION = PROJECT_ROOT / "metrics" / "performance" / req['function']
        return True
    return False


# Evidence "model" in "performance"
def model_available(selected_experiment: str) -> bool :
    global MODEL

    if (PROJECT_ROOT / "experiments" / selected_experiment / "model.pkl").exists():
        with open(PROJECT_ROOT / "experiments" / selected_experiment / "model.pkl", "rb") as f:
            MODEL = pickle.load(f)
        return True
    return False


# Checker
def checker(selected_experiment: str) -> str:
    steps = [
        ("Threshold level defined", threshold_level_is_defined),
        ("Metric measurement available", metric_measurement_available),
        ("Model available", lambda: model_available(selected_experiment)),
        ("Test dataset available", lambda: test_data_set_available(selected_experiment)),
        ("Performance metric computed", f1_score_measure),
        ("Threshold acceptance verified", verifying_acceptance_threshold),
    ]
    string = ""
    all_passed = True
    for name, step in steps:
        try:
            result = step()
        except Exception as e:
            string += f"[FAILED] {name} -> Exception: {e}</br>"
            return string
        if result:
            string += f"[OK] {name}</br>"
        else:
            string += f"[FAILED] {name}</br>"
            all_passed = False
    if all_passed :
        string += "[ALL PASSED]"
    else :
        string += "[FAILED]"
    return string