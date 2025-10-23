import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Strategy "verify" in "release_model"
def verifying_accuracy_is_acceptable() -> bool:
    if BASE_DIR / "model" / "parameters.json".exists() :
        with open(BASE_DIR / "model" / "parameters.json", "r") as f :
            model_json = json.load(f)
        return model_json['performance_measurments']['accuracy'] > 0.8
    return False
    
# Evidence "model" in "release_model"
def trained_model_is_available() -> bool:
    if (BASE_DIR / "model" / "model.pkl").exists() :
        return True
    return False

# Evidence "dataset" in "release_model"
def test_dataset_is_available() -> bool:
    if (BASE_DIR / "data" / "X_test.csv").exists() :
        if (BASE_DIR / "data" / "y_test.csv").exists() :
            return True
    return False