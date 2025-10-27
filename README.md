# FATES MLOPS

This repository contains various experiments conducted as part of the FATES-MLOps project.

It aims to study the compliance to a set of requirements in the case of ML-integrating apps. 

In this context: 
- The sub-repository "usecase" is what represent the ML-integrating app. 
- The sub-repository "ontologies" contains an ontologie about ML and Fairness that is meant to support the requirements.
- The sub-repository "experimental" is a python app that let the user train models and compare their level regarding various metrics, and their compliance to various requirements.

### Poetry dependencies

Poetry is required to install the python environnement with the right dependencies.

Once you have poetry installed, just use :

```
poetry install --no-root
```

You should now have a ".venv" python with the right dependencies to run any code in this repository.

## experimental

To start the experimental app, use :

```
cd experimental
poetry run streamlit run streamlit_app.py
```

## ontologies

...

## usecase

...