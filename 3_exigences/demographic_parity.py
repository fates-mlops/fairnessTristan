from fairlearn.metrics import demographic_parity_difference

def measurement(y_test, y_pred, sensitive_test):
    return demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_test)