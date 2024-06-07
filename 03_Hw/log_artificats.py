if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

import mlflow
import mlflow.sklearn
import joblib
@data_exporter
def export_data(artifacts):
    
    print('netered')
    model , dv = artifacts
    
    
    
    
    mlflow.set_tracking_uri("http://mlflow:5000")
    print('done')
    
    
    mlflow.set_experiment('mage_log')
    print('set')
    with mlflow.start_run():
        print('run started')    
        mlflow.sklearn.log_model(model,'lr_model')
        dict_vectorizer_path = "dict_vectorizer.pkl"
        joblib.dump(dv, dict_vectorizer_path)
        mlflow.log_artifact(dict_vectorizer_path)
    print('logged succesfully')
    # Specify your data exporting logic here

