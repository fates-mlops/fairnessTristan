from fairlearn.metrics import MetricFrame
from sklearn.metrics import precision_score

def metric(y_test, y_pred, sensitive_test):
    mf = MetricFrame(metrics=precision_score, y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_test)
    return abs(mf.by_group.iloc[0] - mf.by_group.iloc[1])