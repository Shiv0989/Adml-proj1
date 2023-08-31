import pandas as pd
import numpy as np

class DataPreparation:
    def __init__(self, train_path, test_path):
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)
        
    def set_display_options(self):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        
    def check_null_values(self):
        null_counts = self.train_data.isnull().sum()
        return null_counts
    
    def extract_features_and_target(self):
        y = self.train_data['drafted']
        X = self.train_data.drop(['drafted', 'player_id'], axis=1)
        test_player_ids = self.test_data['player_id'].copy()
        self.test_data = self.test_data.drop('player_id', axis=1)
        return X, y, test_player_ids
    
    def handle_missing_values(self, X):
        numerical_cols = X.select_dtypes(include=np.number).columns
        for col in numerical_cols:
            median_value = X[col].median()
            X[col].fillna(median_value, inplace=True)
            self.test_data[col].fillna(median_value, inplace=True)
        return X
