from sklearn.metrics import accuracy_score

def metric(y_test, y_pred):
    return accuracy_score(y_test, y_pred)