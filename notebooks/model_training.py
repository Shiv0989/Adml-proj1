from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

class ModelTraining:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def split_and_balance_data(self):
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        return X_train_smote, y_train_smote, X_val, y_val
    
    def train_model(self, model):
        model.fit(self.X, self.y)
        return model
