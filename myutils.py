from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def acc_score(X_train, X_test, y_train, y_test, model, model_name):
    y_test_model = model.predict(X_test)
    y_train_model = model.predict(X_train)

    acc_train_model = accuracy_score(y_train, y_train_model)
    acc_test_model = accuracy_score(y_test, y_test_model)
    
    print("{}: Accuracy on training Data: {:.3f}".format(model_name, acc_train_model))
    print("{}: Accuracy on test Data: {:.3f}".format(model_name, acc_test_model))

    return 

def acc_score4xgb(X_train, X_test, y_train, y_test, model):
    le = LabelEncoder()
    y_train_xgb = le.fit_transform(y_train)
    y_test_xgb = le.transform(y_test)

    #predicting the target value from the model for the samples
    y_test_xgb_1 = model.predict(X_test)
    y_train_xgb_1 = model.predict(X_train)

    #computing the accuracy of the model performance
    acc_train_model = accuracy_score(y_train_xgb,y_train_xgb_1)
    acc_test_model = accuracy_score(y_test_xgb,y_test_xgb_1)

    print("XGBoost: Accuracy on training Data: {:.3f}".format(acc_train_model))
    print("XGBoost : Accuracy on test Data: {:.3f}".format(acc_test_model))

    return 

def save_model(model, path):
    import pickle
    with open(path, 'wb') as file:
        pickle.dump(model, file)

    return 