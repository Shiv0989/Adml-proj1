from sklearn.metrics import roc_auc_score
import pandas as pd

class PredictModel:
    def __init__(self, model, X_val=None, y_val=None):
        self.model = model
        self.X_val = X_val
        self.y_val = y_val

        if X_val is not None and y_val is not None:
            if X_val.empty or y_val.empty:
                raise ValueError("X_val or y_val is empty, cannot proceed.")
            
            if X_val.isnull().any().any() or y_val.isnull().any():
                raise ValueError("X_val or y_val contains NaN values, cannot proceed.")

    def evaluate_model(self):
        if self.X_val is None or self.y_val is None:
            raise ValueError("X_val or y_val is None, cannot evaluate the model.")

        val_predictions = self.model.predict_proba(self.X_val)[:, 1]
        val_auroc = roc_auc_score(self.y_val, val_predictions)
        return val_auroc

    def generate_submission(self, X_test, test_player_ids, model_name):
        # Generate predictions
        print("Columns in X_val but not in X_test:", set(self.X_val.columns) - set(X_test.columns))
        print("Columns in X_test but not in X_val:", set(X_test.columns) - set(self.X_val.columns))

        kaggle_predictions = self.model.predict_proba(X_test)[:, 1]
        
        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'player_id': test_player_ids,
            'probability': kaggle_predictions
        })

        submission_df.to_csv(f'{model_name}_submission.csv', index=False)
