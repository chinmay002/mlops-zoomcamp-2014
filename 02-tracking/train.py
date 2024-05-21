import os
import pickle
import click
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

data_path = 'C:\\Users\\Chinmay\\Desktop\\Docker'

def load_pickle(filename):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)

def train_model(data_path):
    mlflow.set_tracking_uri(uri = '')
    mlflow.set_experiment('new_exp')

    mlflow.autolog(log_models=True)
    
    with mlflow.start_run():
        X_train,y_train = load_pickle(os.path.join(data_path, 'train.pkl'))
        X_val,y_val = load_pickle(os.path.join(data_path, 'val.pkl'))
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        print(rmse)
    
    


if __name__ == '__main__':
    train_model(data_path)