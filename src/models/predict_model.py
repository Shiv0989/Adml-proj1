from sklearn.metrics import roc_auc_score
import pandas as pd

class PredictModel:
    def __init__(self, model, X_val=None, y_val=None):
        self.model = model
        self.X_val = X_val
        self.y_val = y_val

    def evaluate_model(self):
        val_predictions = self.model.predict_proba(self.X_val)[:, 1]
        val_auroc = roc_auc_score(self.y_val, val_predictions)
        return val_auroc
    
    def predict_on_test(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]  # Assuming your model has a predict_proba method

