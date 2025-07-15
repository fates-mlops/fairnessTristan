import pandas as pd

def preprocess(dataset, experiment_path) :
    # Missing values replacement
    dataset.replace('?', pd.NA, inplace=True)
    for col in dataset.columns:
        dataset[col] = dataset[col].fillna('Unknown')

    # From qualitative to binary
    dataset['sex'] = dataset.sex == 'Male'
    dataset['income'] = dataset.income == '>50K'
    dataset = pd.get_dummies(dataset, columns=['workclass', 'marital.status', 'occupation', 'relationship', 'native.country'])

    # Type transformation
    int_cols = dataset.select_dtypes(include="int").columns
    dataset[int_cols] = dataset[int_cols].astype("float64")

    # Agregated feature
    dataset['capital_diff'] = dataset['capital.gain'] - dataset['capital.loss']

    # Columns to be removed
    dataset = dataset.drop(columns=['race', 'fnlwgt', 'education', 'capital.gain', 'capital.loss'])

    dataset.to_csv(f"{experiment_path}/preprocessed_data.csv", index=None)
    return dataset