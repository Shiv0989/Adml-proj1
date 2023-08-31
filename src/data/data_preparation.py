import pandas as pd
import numpy as np

class DataPreparation:
    def __init__(self, train_path, test_path):
        self.train_data = pd.read_csv(train_path, low_memory=False)
        self.test_data = pd.read_csv(test_path, low_memory=False)

    def handle_missing_values(self, df, is_train=True):
        # Handle missing numerical values
        numerical_cols = df.select_dtypes(include=np.number).columns
        for col in numerical_cols:
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
        
        # Handle missing categorical values
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            mode_value = df[col].mode()[0]
            df[col].fillna(mode_value, inplace=True)
        
        # For training data, we also one-hot encode the categories
        if is_train:
            df = pd.get_dummies(df, columns=categorical_cols)
        
        return df

    def align_columns(self):
        # Make sure the test set has the same columns as the training set
        missing_cols = set(self.train_data.columns) - set(self.test_data.columns)
        
        # Convert the set to a list
        missing_cols_list = list(missing_cols)
        
        # Create a new DataFrame for missing columns
        missing_df = pd.DataFrame(0, index=np.arange(len(self.test_data)), columns=missing_cols_list)
        
        # Concatenate the missing columns to the original DataFrame
        self.test_data = pd.concat([self.test_data, missing_df], axis=1)
        
        self.test_data = self.test_data[self.train_data.columns]


    
    def handle_missing_and_categorical_values(self):
        self.train_data = self.handle_missing_values(self.train_data, is_train=True)
        self.test_data = self.handle_missing_values(self.test_data, is_train=False)
        self.align_columns()
        
    def extract_features_and_target(self):
        drop_cols = ['drafted']
        if 'player_id' in self.train_data.columns:
            drop_cols.append('player_id')
        y = self.train_data['drafted'].copy()
        X = self.train_data.drop(drop_cols, axis=1).copy()
        test_player_ids = self.test_data['player_id'].copy() if 'player_id' in self.test_data.columns else None
        X_test = self.test_data.drop('player_id', axis=1).copy() if 'player_id' in self.test_data.columns else self.test_data.copy()
        return X, y, test_player_ids, X_test

