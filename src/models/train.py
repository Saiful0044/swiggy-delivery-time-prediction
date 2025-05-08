import pandas as pd
import yaml
import joblib
import logging
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestRegressor,StackingRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from pathlib import Path

target_col = 'time_taken'
# create logger
logger = logging.getLogger("model_training")
logger.setLevel(logging.INFO)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# add handler to logger
logger.addHandler(handler)

# create a fomratter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to handler
handler.setFormatter(formatter)


# Data loading function
def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logger.info("File loaded successfully")
        return df
    except FileNotFoundError:
        logger.error(f"The file to load does not exist: {data_path}")
        raise

# X,y split method
def make_X_and_y(data: pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X,y


# parameters read
def read_params(file_path:Path) -> dict:
    with open(file_path,'r') as f:
        params_file = yaml.safe_load(f)
    return params_file

# model fit method
def train_model(model,X_train: pd.DataFrame,y_train: pd.Series):
    model.fit(X_train,y_train)
    return model

# model save
def save_model(model, save_dir: Path,model_name: str):
    save_location = save_dir/model_name
    # save the model
    joblib.dump(value=model, filename=save_location)

# save transformer
def save_transformer(transformer, save_dir: Path, transformer_name: str):
    save_location = save_dir / transformer_name
    joblib.dump(value=transformer, filename=save_location)




# Main block
if __name__ == '__main__':
    root_path = Path(__file__).resolve().parents[2]
    data_path = root_path / "data" / "processed" / "train_trans.csv"
    # parameters file
    params_file_path = root_path / "params.yaml"

    # load the training data
    training_data = load_data(data_path)

    # split the data into X and y
    X_train,y_train = make_X_and_y(training_data, target_col)
    logger.info("Dataset splitting completed")

    # model parameters
    model_params = read_params(params_file_path)['Train']

    # rf params
    rf_params = model_params['Random_Forest']
    logger.info("Random forest parameters read")

    # build random forest model
    rf = RandomForestRegressor(**rf_params)
    logger.info("Build random forest model")

    # light gbm params
    light_params = model_params['LightGBM']
    logger.info("Light GBM parameters read")

    lgbm = LGBMRegressor(**light_params)
    logger.info("Build Light GBM model")
    
    # meta model
    lr = LinearRegression()
    logger.info("Meta model built")

    # power transformer
    power_transform = PowerTransformer()
    logger.info("Target Power Transformer built")

    # form the stacking regressor
    stacking_reg = StackingRegressor(estimators=[
        ('rf_model', rf),
        ('lgbm_model', lgbm)], 
    final_estimator=lr,
    cv=5, n_jobs=-1)

    logger.info('Stacking Regressor Built')

    # make the model wraper
    model = TransformedTargetRegressor(
        regressor=stacking_reg,
        transformer=power_transform)
    logger.info("Models wrapped inside wrapper")

    # fit the model on training data
    train_model(model=model, X_train=X_train, y_train=y_train)
    logger.info('Model Training Completed')

    # model name 
    model_filename = "model.joblib"
    model_save_dir = root_path / "models"
    model_save_dir.mkdir(exist_ok=True)

    # save the model
    save_model(model=model,
               save_dir=model_save_dir, 
               model_name=model_filename)
    logger.info('Trained model saved to location')


    # extract the model from wrapper
    stacking_model = model.regressor_
    transformer = model.transformer_

    # save the stacking model
    stacking_filename = 'stacking_regressor.joblib'
    save_model(model=stacking_model,
               save_dir=model_save_dir,
               model_name=stacking_filename)
    logger.info("Stacking model save to location")

    # save the transformer
    transformer_filename = 'power_transformer.joblib'
    transformer_save_dir = model_save_dir
    save_transformer(transformer=transformer,
                     save_dir=transformer_save_dir,
                     transformer_name=transformer_filename
                    )
    logger.info("Transformer save to location")





