from sklearn.metrics import roc_auc_score

class PredictModel:
    def __init__(self, model, X_val, y_val):
        self.model = model
        self.X_val = X_val
        self.y_val = y_val

    def evaluate_model(self):
        val_predictions = self.model.predict_proba(self.X_val)[:, 1]
        val_auroc = roc_auc_score(self.y_val, val_predictions)
        return val_auroc
