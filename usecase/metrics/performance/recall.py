from sklearn.metrics import recall_score

def metric(y_test, y_pred):
    return recall_score(y_test, y_pred)