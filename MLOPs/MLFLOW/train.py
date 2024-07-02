import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import mlflow


def evaluate_metric(y_pred,Y_true):
    rmse = mean_squared_error(Y_true,y_pred,squared=False)
    mae = mean_absolute_error(Y_true,y_pred)
    r2 = r2_score(Y_true,y_pred)
    
    return(rmse,mae,r2)


if __name__ == "__main__":
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    data = pd.read_csv(csv_url, sep=";")
    X = data.drop(columns="quality")
    y = data['quality']
    
    X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=0)
    with mlflow.start_run():
        alpha=0.5
        l1_ratio = 0.6
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=0)
        lr.fit(X_train,y_train)
        
        prediction = lr.predict(X_test)
        
        (rmse,mae,r2) = evaluate_metric(prediction,y_test)
        print(f"rmse is {rmse}")
        print(f"mae is {mae}")
        print(f"r2 is {r2}")
        
        mlflow.log_param("alpha",alpha)
        mlflow.log_param("l1_ratio",l1_ratio)
        mlflow.log_metric("rmse",rmse)
        mlflow.log_metric("mae",mae)
        mlflow.log_metric("r2",r2)
        
        mlflow.sklearn.log_model(lr,"model")
        mlflow.log_artifact("file.txt")
        