import numpy as np
import pandas as pd
from scipy import stats

class DataPreparation:
    def __init__(self, train_path, test_path):
        self.train_data = pd.read_csv(train_path, low_memory=False)
        self.test_data = pd.read_csv(test_path, low_memory=False)

        # Save player_id and drafted columns for later use
        self.train_player_ids = self.train_data['player_id'].copy()
        self.test_player_ids = self.test_data['player_id'].copy()
        self.y = self.train_data['drafted'].copy()


    def handle_missing_values(self, df, one_hot_encode=False):
        numerical_cols = df.select_dtypes(include=np.number).columns
        for col in numerical_cols:
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)

        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            mode_value = df[col].mode()[0]
            df[col].fillna(mode_value, inplace=True)
        
        if one_hot_encode:
            # Remove 'player_id' from the list of columns to be one-hot encoded
            cols_to_encode = [col for col in categorical_cols if col != 'player_id']
            df = pd.get_dummies(df, columns=cols_to_encode)
        return df



    def extract_features_and_target(self):
        drop_columns = ['player_id', 'drafted']
        X = self.train_data.drop(drop_columns, axis=1, errors='ignore')
        X_test = self.test_data.drop(['player_id'], axis=1)
           
        y = self.y  # self.y already holds the 'drafted' column
        test_player_ids = self.test_player_ids  # Already saved in __init__

        return X, y, test_player_ids, X_test


    def feature_engineering(self, df):
        if "birth_year" in df.columns:
            df["age"] = 2023 - df["birth_year"]

        df["shooting_efficiency"] = np.where((df["twoPA"] + df["TPA"]) != 0, df["pts"] / (df["twoPA"] + df["TPA"]), 0)
        df["rebound_efficiency"] = np.where((df["oreb"] + df["dreb"]) != 0, df["treb"] / (df["oreb"] + df["dreb"]), 0)
        return df
    
    def check_for_nans(self, df):
        if df.isnull().any().any():
            raise ValueError("DataFrame contains NaN values.")

    def transform_year(self, df):
        df["year_sin"] = np.sin(2 * np.pi * (df["year"] - 1951) / 72)
        df["year_cos"] = np.cos(2 * np.pi * (df["year"] - 1951) / 72)
        
        return df

    def preprocess_data(self):
        self.train_data = self.handle_missing_values(self.train_data, one_hot_encode=True)
        self.test_data = self.handle_missing_values(self.test_data, one_hot_encode=True)
        
        #self.extract_features_and_target()

        self.train_data = self.feature_engineering(self.train_data)
        self.train_data = self.transform_year(self.train_data)
        self.check_for_nans(self.train_data)

        self.test_data = self.feature_engineering(self.test_data)
        self.test_data = self.transform_year(self.test_data)
        self.check_for_nans(self.test_data)
