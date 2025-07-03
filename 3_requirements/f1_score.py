from sklearn.metrics import f1_score

def measurement(y_test, y_pred):
    return f1_score(y_test, y_pred)