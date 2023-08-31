from src.data.data_preparation import DataPreparation
from src.models.train_model import ModelTraining
from src.models.predict_model import ModelPrediction
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    train_data_path = "data/raw/train.csv"
    test_data_path = "data/raw/test.csv"
    
    # Data Preparation
    data_prep = DataPreparation(train_data_path, test_data_path)
    data_prep.set_display_options()
    print("Null values in each column:", data_prep.check_null_values())
    
    X, y, test_player_ids = data_prep.extract_features_and_target()
    X = data_prep.handle_missing_values(X)
    
    # Model Training
    model_training = ModelTraining(X, y)
    X_train_smote, y_train_smote, X_val, y_val = model_training.split_and_balance_data()
    
    model = RandomForestClassifier()
    trained_model = model_training.train_model(model)
    
    # Model Prediction and Evaluation
    model_prediction = ModelPrediction(trained_model, X_val, y_val)
    val_auroc = model_prediction.evaluate_model()
    print(f"Validation AUROC Score: {val_auroc}")
    
    # Add code for making predictions on the test set and other tasks
