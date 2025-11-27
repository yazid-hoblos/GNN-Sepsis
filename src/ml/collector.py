class ResultsCollector:
    """
    Minimal collector to store predictions for plotting.
    Stores y_test, y_pred, y_proba per model and input.
    """
    def __init__(self):
        self.data = {}  # key = (model_name, input_name), value = (y_test, y_pred, y_proba)

    def add(self, model_name, input_name, y_test, y_pred, y_proba):
        self.data[(model_name, input_name)] = (y_test, y_pred, y_proba)

    def get(self, model_name, input_name):
        return self.data.get((model_name, input_name), (None, None, None))
