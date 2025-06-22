# Use cases by Tristan

## Validation of compliance with requirements

Create the python virtual env

```
python -m venv .venv
```

Start the python virtual env

```
./.venv/Scripts/activate
```

Install dependencies

```
pip install -r requirements.txt
```

Start the app

```
streamlit run streamlit_app.py
```

## UseCases

### 1_data

Contient les données utilisées.

#### 1_raw

Contient la base de donnée originale et une explication sur son origine.

#### 2_preprocessed

Contient les données nettoyées, et une éventuelle indication sur la stratégie d'échantillonnage.

#### 3_split

Contient les données séparées pour l'entrainement et le test et une indication sur les paramètres utilisés pour cette séparation.

### 2_models

Contient le modèle entrainé et des indications sur ses hyperparamètres.

### 3_notebooks

Contient le code python
