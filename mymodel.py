# Description: This file contains the code for the model training and evaluation.
# creator: Tian
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.svm import SVC

def DECISION_TREE(X_train, y_train, _max_depth=8):
    # instantiate the model 
    tree = DecisionTreeClassifier(max_depth = _max_depth)
    # fit the model 
    tree.fit(X_train, y_train)

    return tree

def RANDOM_FOREST(X_train, y_train, _max_depth=15):
    # instantiate the model
    forest = RandomForestClassifier(max_depth=_max_depth)

    # fit the model 
    forest.fit(X_train, y_train)

    return forest

def MLP(X_train, y_train, _alpha=0.001, _hidden_layer_sizes=([100,100,100])):
    # instantiate the model
    mlp = MLPClassifier(alpha=_alpha, hidden_layer_sizes=_hidden_layer_sizes)

    # fit the model 
    mlp.fit(X_train, y_train)

    return mlp


def XGBOOST(X_train, y_train, _learning_rate=0.4, _max_depth=7):
    # 编码目标变量 y
    le = LabelEncoder()
    y_train_xgb = le.fit_transform(y_train)

    # 创建并训练 XGBoost 模型
    xgb = XGBClassifier(learning_rate=_learning_rate,max_depth=_max_depth)
    xgb.fit(X_train, y_train_xgb)

    return xgb

def SVM(X_train, y_train, _kernel='linear', _C=1.0, _random_state=12):
    # instantiate the model
    svm = SVC(kernel=_kernel, C=_C, random_state=_random_state)
    #fit the model
    svm.fit(X_train, y_train)

    return svm

