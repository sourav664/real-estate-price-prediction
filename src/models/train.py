import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import logging
import yaml
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor , StackingRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import TransformedTargetRegressor
from lightgbm import LGBMRegressor

TARGET = 'price'

#create logger
logger = logging.getLogger("train_model")
logger.setLevel(logging.DEBUG)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# file handler
file_handler = logging.FileHandler('train_model.log')
file_handler.setLevel(logging.ERROR)

# add handler to logger
logger.addHandler(handler)

# create a fomratter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to handler
handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        
    except FileNotFoundError:
        logger.error("The file to load does not exist")
        raise
        
    return df


def read_params(file_path):
    with open(file_path, 'r') as f:
        params_file = yaml.safe_load(f)
        
    return params_file

def save_model(model, save_dir: Path, model_name: str) -> None:
    # form the save location
    save_location = save_dir / model_name
    
    # save the model
    joblib.dump(value=model, filename=save_location)


def save_transformer(transformer, save_dir: Path, transformer_name: str) -> None:
    # form the save location
    save_location = save_dir / transformer_name
    
    # save the transformer
    joblib.dump(value=transformer, filename=save_location)
    
    
def train_model(model, X_train: pd.DataFrame, y_train):
    # fit the model
    model.fit(X_train, y_train)
    
    return model

def make_X_and_y(data: pd.DataFrame, target: str) -> tuple:
    X = data.drop(columns=[target])
    y = data[target]
    
    return X, y


def target_value_transform(train_y: pd.DataFrame, test_y: pd.DataFrame) -> tuple:
    
    # create a transformer for target value using functiontransformer
    transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True)
    
    # transform the target value
    train_y_transformed = transformer.fit_transform(train_y.values.reshape(-1, 1))
    test_y_transformed = transformer.transform(test_y.values.reshape(-1, 1))
    
    return train_y_transformed, test_y_transformed

if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent
    # params path
    params_file_path = root_path / "params.yaml"
    
        
    # data path
    train_transformed_path = root_path / "data" / "processed" / "train_trans.csv"
   
    
    # read the data
    train_transformed = load_data(train_transformed_path)
    logger.info("Training data read successfully")
    
    
    # split the transformed data
    X_train, y_train = make_X_and_y(train_transformed, TARGET)
    logger.info("Training data split successfully")
    
    
    # model parameters
    model_params = read_params(params_file_path)['Train']
    
    
   
    
    # rf_params
    rf_params = model_params['Random_Forest']
    logger.info("random forest parameters read")
    
    # build random forest model
    rf = RandomForestRegressor(**rf_params)
    logger.info("built random forest model")
    
    # light gbm params
    lgbm_params = model_params["LightGBM"]
    logger.info("Light GBM parameters read")
    lgbm = LGBMRegressor(**lgbm_params)
    logger.info("built Light GBM model")
    
    # meta model
    lr = LinearRegression()
    logger.info("Meta model built")
    
    # log transformer
    log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True)
    logger.info("Target Transformer built")
    
    
    # form the stacking regressor
    stacking_reg = StackingRegressor(estimators=[("rf_model",rf),
                                                 ("lgbm_model",lgbm)],
                                     final_estimator=lr,
                                     cv=5,n_jobs=-1)
    logger.info("Stacking regressor built")
    
    # make the model wrapper
    model = TransformedTargetRegressor(regressor=stacking_reg,
                                       transformer=log_transformer)
    logger.info("Model wrapper built")
    
    # fit the model on training data
    train_model(model, X_train, y_train)
    logger.info("Model trained on training data")
    
    # model filename
    model_filename = "model.joblib"
    
    # directory to save the model
    model_save_dir = root_path / "models"
    model_save_dir.mkdir(exist_ok=True)
    
    
    # extract the model from wrapper
    stacking_model = model.regressor_
    transformer = model.transformer_
    
    # save the model
    save_model(model=model,
            save_dir=model_save_dir,
            model_name=model_filename)
    logger.info("Trained model saved to location")
    
    # save the stacking model
    stacking_filename = "stacking_regressor.joblib"
    save_model(model=stacking_model,
            save_dir=model_save_dir,
            model_name=stacking_filename)
    logger.info("Trained model saved to location")
    
    # save the transformer
    transformer_filename = "log_transformer.joblib"
    transformer_save_dir = model_save_dir
    save_transformer(transformer=transformer,
                    save_dir=transformer_save_dir,
                    transformer_name=transformer_filename)
    logger.info("Target transformer saved to location")
    