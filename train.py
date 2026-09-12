import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


def read_define(filepath, y_col):
    df= pd.read_csv(filepath)
    X=df.drop(y_col, axis=1)
    y=df[y_col]

    return X,y

def train_time(X,y):
    X_train,X_test,y_train,y_test= train_test_split(
        X,
        y,
        test_size=.2,
        random_state=42,
        stratify=y)
    return X_train,X_test,y_train,y_test    

def modelling(X_train,X_test,y_train,y_test) :
    model=LogisticRegression(max_iter=1000)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    y_proba=model.predict_proba(X_test)
    return y_pred ,y_proba        # pehli 5 probabilities

def conf_matrix(y_actual, y_pred):
    cm=confusion_matrix(y_actual, y_pred)
    return cm

read_X,read_y=read_define("creditcard.csv", "Class")
X_train,X_test,y_train,y_test=train_time(read_X,read_y)
model_pred,model_prob=modelling(X_train,X_test,y_train,y_test)    
evalu=conf_matrix(y_test,model_pred)

print(model_pred[:20])
#print(model_prob[:5])
print(evalu)
print(classification_report(y_test, model_pred)) # gives all the evalution metrics