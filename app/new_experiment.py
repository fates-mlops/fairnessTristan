import streamlit as st
import sys
import os
import json

dataset_preprocessing = {}
data_dir = "./data"
for dataset_name in os.listdir(data_dir) :
    dataset_path = os.path.join(data_dir, dataset_name)
    if os.path.isdir(dataset_path) :
        preprocessing_dir = os.path.join(dataset_path, "preprocessing")
        if os.path.isdir(preprocessing_dir) :
            scripts = [os.path.splitext(f)[0] for f in os.listdir(preprocessing_dir) if f.endswith(".py")]
            dataset_preprocessing[dataset_name] = scripts

models = {
    'classification' : [
        {
            'name' : 'Random Forest Classifier',
            'hyperparameters' : ['n_estimators', 'max_depth', 'random_state']
        }
    ]
}

performance_path = "./metrics/performance"
performance_metrics_list = [file[:-3] for file in os.listdir(performance_path) if file.endswith('.py') and os.path.isfile(os.path.join(performance_path, file))]
fairness_path = "./metrics/fairness"
fairness_metrics_list = [file[:-3] for file in os.listdir(fairness_path) if file.endswith('.py') and os.path.isfile(os.path.join(fairness_path, file))]
fairness_treatment_list = [
    "undersampling_class_balance", 
    "hide_protected"
]
fairness_treatment_param_list = {
    "undersampling_class_balance" : ["random_state"],
    "hide_protected" : []
}

def app():

    st.title("New Experiment")
    is_valid = True

    # Fill the experiment name
    experiment_name = st.text_input("Experiment name")
    existing_experiments = [name for name in os.listdir("experiments") if os.path.isdir(os.path.join("experiments", name))]
    if experiment_name in existing_experiments :
        st.warning("This name is already taken. Please choose another experiment name.")
        is_valid = False
    
    st.subheader("Data Preparation")
    # Choose the data source
    dataset_options = ["Choose a dataset"] + list(dataset_preprocessing.keys())
    data_source = st.selectbox("Dataset", dataset_options)
    if data_source == "Choose a dataset" :
        st.warning("Please choose a dataset")
        is_valid = False

    # Choose the data preprocessing
    if data_source != "Choose a dataset" :
        preprocessing_options = ["Choose a preprocessing"] + list(dataset_preprocessing[data_source])
        preprocessing = st.selectbox("Data Preprocessing", preprocessing_options)
    else :
        preprocessing = None
    if preprocessing == "Choose a preprocessing" :
        st.warning("Please choose a preprocessing")
        is_valid = False

    
    st.subheader("Model Preparation")
    # Choose the type of ML process
    learning_task_options = ["Choose a learning task"] + list(models.keys())
    learning_task = st.selectbox("Learning task", learning_task_options)
    if learning_task == "Choose a learning task" :
        st.warning("Please choose a learning task")
        is_valid = False
        model_type = None
        hyperparameters = None
    else :
    # Choose the type of model
        model_type_options = ["Choose a type of model"] + list(m['name'] for m in models[learning_task])
        model_type = st.selectbox("Model type", model_type_options)
        # Select the hyperparameters
        if model_type != "Choose a type of model" :
            model_type_hyperparameters = next((m['hyperparameters'] for m in models[learning_task] if m['name'] == model_type), [])
            hyperparameters = {}
            for hyperparameter in model_type_hyperparameters :
                hyperparameters[hyperparameter] = st.number_input(f"Hyperparameter: {hyperparameter}") 
                # Séparer les hyperparamètres par types de données pour améliorer la validation dans le formulaire.

    # Choose the target
    if learning_task == "classification" :
        target = st.text_input("Target feature")
    else :
        target = None
    if target == "" :
        st.warning("Please choose a target feature")
        # Mettre en place une validation sur les colonnes existantes -> Possible sans effectuer le prétraitement dans le vide ?
        is_valid = False

    # Split parameters : Test size & random seed
    test_size = st.number_input("Test sample size", min_value=0.0, max_value=1.0, step=0.01)
    if test_size <= 0 or test_size >= 1 :
        st.warning("Please choose a test size between 0 and 1 (excluded)")
        is_valid = False
    split_seed = st.number_input("Train & Test split random seed", min_value=0, max_value=100, step=1)

    st.subheader("Performance Monitoring")
    performance_metrics = []
    for metric in performance_metrics_list :
        metric_check = st.checkbox(f"Use metric: {metric}")
        if metric_check : 
            performance_metrics.append(metric)
    
    st.subheader("Fairness Monitoring")
    # Fairness : Protected feature
    fairness = st.checkbox("Fairness: Protect a sub-group")
    if fairness:
        protected = st.text_input("Protected feature")
    else :
        protected = None
    if protected == "" :
        st.warning("Please choose a protected feature")
        # Mettre en place une validation sur les colonnes existantes -> Possible sans effectuer le prétraitement dans le vide ?
        is_valid = False

    fairness_metrics = []
    fairness_treatment = []
    fairness_treatment_param = {}
    if fairness:
        for metric in fairness_metrics_list :
            metric_check = st.checkbox(f"Use fairness mertric: {metric}")
            if metric_check :
                fairness_metrics.append(metric)
        for treatment in fairness_treatment_list :
            treatment_check = st.checkbox(f"Use fairness treatment: {treatment}")
            if treatment_check :
                fairness_treatment.append(treatment)
                fairness_treatment_param[treatment] = {}
                for param in fairness_treatment_param_list[treatment] :
                    fairness_treatment_param[treatment][param] = st.number_input(param)
    

    if st.button("Send"):
        if is_valid :
            param_json = {}
            param_json["experiment_name"] = experiment_name
            param_json["data_source"] = {
                "dataset" : data_source,
                "preprocessing" : preprocessing
            }
            param_json["split_param"] = {
                "target" : target,
                "protected" : protected,
                "test_size" : test_size,
                "seed" : split_seed
            }
            param_json["model"] = {
                "type" : model_type,
                "hyperparameters" : hyperparameters
            }
            param_json["performance_metrics"] = performance_metrics
            param_json["fairness_metrics"] = fairness_metrics
            param_json["fairness_treatment"] = fairness_treatment
            param_json["fairness_treatment_param"] = fairness_treatment_param
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
            from experiment import run_experiment
            run_experiment(param_json)
            st.write("Experiment successfully completed.")

        else :
            st.error("Could not start the experiment. Please fill the requested parameters.")