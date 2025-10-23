from fairlearn.metrics import demographic_parity_ratio

def metric(y_test, y_pred, sensitive_test):
    return demographic_parity_ratio(y_test, y_pred, sensitive_features=sensitive_test)