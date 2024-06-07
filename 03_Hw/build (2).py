from mlops.utils.data_preparation.encoders import vectorize_features


if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_exporter
def export(data) :
    
    
    df = data.iloc[:,:]
    target = 'duration'


    print('bbb')
    X_train, dv = vectorize_features(df[['PULocationID','DOLocationID']])
    y_train = df[target]    

    print('done')

    
   
    return  X_train,y_train, dv

@test
def test_output(X_train,y_train, dv):
    assert X_train is not None
    assert y_train is not None
    assert dv is not None

