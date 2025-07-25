from fairlearn.metrics import equalized_odds_difference

def metric(y_test, y_pred, sensitive_test):
    return equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_test)