from sklearn.metrics import precision_score

def metric(y_test, y_pred):
    return precision_score(y_test, y_pred)