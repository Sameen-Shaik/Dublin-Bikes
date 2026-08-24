import mlflow
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from main import prepare_data, train_model
from src.config import LGB_PARAMS, XGB_PARAMS
from src.models import build_preprocessor, evaluate_predictions

# Enable autologging - captures everything automatically
mlflow.autolog()

models_list = [ XGBRegressor, LGBMRegressor]
params_list = [ XGB_PARAMS, LGB_PARAMS]


#Select a model to train and track
model = models_list[0]
params = params_list[0]

def run_experiment():
    print("Loading data...")
    X_train_val, y_train_val, X_test, y_test, numeric_features, categorical_features, test_df = prepare_data()
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # Train model - MLflow automatically logs everything!
    with mlflow.start_run(run_name=f"{model.__name__} Trail"):
        preprocessor = build_preprocessor(numeric_features, categorical_features)
        X_train_preprocessed = preprocessor.fit_transform(X_train_val)
        X_test_preprocessed = preprocessor.transform(X_test)

        final_pipeline = train_model(model, params, preprocessor, X_train_preprocessed, y_train_val, X_test_preprocessed, y_test, test_df, numeric_features, categorical_features)

        # Evaluate on test set
        y_pred = final_pipeline.predict(X_test)
        metrics = evaluate_predictions("test", y_test, y_pred)
        del metrics["model"]
        mlflow.log_metrics(metrics)

if __name__ == "__main__":
    run_experiment()