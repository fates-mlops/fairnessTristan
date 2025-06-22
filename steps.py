import json
from pathlib import Path

# Strategy "verify" in "release_model"
def verifying_accuracy_is_acceptable() -> bool:
    if Path('./2_models/random_forest_model.json').exists() :
        with open('./2_models/random_forest_model.json', "r") as f:
            model_json = json.load(f)
        return model_json['metriques']['accuracy'] > 0.8
    return False
    
# Evidence "model" in "release_model"
def trained_model_is_available() -> bool:
    if Path("./2_models/random_forest_model.pkl").exists():
        return True
    return False

# Evidence "dataset" in "release_model"
def test_dataset_is_available() -> bool:
    if Path("./1_data/3_split/X_test.csv").exists():
        if Path("./1_data/3_split/y_test.csv").exists():
            return True
    return False