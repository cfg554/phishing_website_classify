from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
import mymodel
import argparse
import myutils
  
# fetch dataset 
phishing_websites = fetch_ucirepo(id=379) 
  
# data (as pandas dataframes) 
# phishing_websites.data.features为X，phishing_websites.data.targets为y，都是pandas的DataFrame

data = pd.concat([phishing_websites.data.features, phishing_websites.data.targets ], axis=1)

# shuffle
data = data.sample(frac=1).reset_index(drop=True)

y = data['Result']
X = data.drop('Result',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, random_state = 12)

model_names = sorted(name for name in mymodel.__dict__
    if name.isupper() and callable(mymodel.__dict__[name])
    and name != 'SVC')

print(f'Available models: {model_names}')
# train the model
parser = argparse.ArgumentParser(description='choose the model and parameters for model training')

parser.add_argument('--arch', type=str, default='XGBOOST', help='choose the model to train', choices=model_names)

parser.add_argument('--decision_tree_depth', type=int, default=8, help='max_depth for decision tree')

parser.add_argument('--random_forest_depth', type=int, default=15, help='max_depth for random forest')

parser.add_argument('--mlp_alpha', type=float, default=0.001, help='alpha for MLP')

parser.add_argument('--mlp_hidden_layer_sizes', type=int, nargs='+', default=[100,100,100], help='hidden_layer_sizes for MLP')

parser.add_argument('--xgboost_learning_rate', type=float, default=0.4, help='learning_rate for XGBOOST')

parser.add_argument('--xgboost_max_depth', type=int, default=7, help='max_depth for XGBOOST')

parser.add_argument('--svc_kernel', type=str, default='rbf', help='kernel for SVC', choices=['linear', 'poly', 'rbf', 'sigmoid'])

parser.add_argument('--svc_C', type=float, default=1.0, help='C for SVC')

parser.add_argument('--svc_random_state', type=int, default=12, help='random_state for SVC')

parser.add_argument('--save-dir', type=str, default='model.pkl', help='path to save the model')

args = parser.parse_args()


def main():
    #训练模型
    my_model = None
    print(f'model structure you choose is {args.arch}')
    if args.arch == 'DECISION_TREE':
        my_model = mymodel.DECISION_TREE(X_train, y_train, _max_depth=args.decision_tree_depth)
    elif args.arch == 'RANDOM_FOREST':
        my_model = mymodel.RANDOM_FOREST(X_train, y_train, _max_depth=args.random_forest_depth)
    elif args.arch == 'MLP':
        my_model = mymodel.MLP(X_train, y_train, _alpha=args.mlp_alpha, _hidden_layer_sizes=args.mlp_hidden_layer_sizes)
    elif args.arch == 'XGBOOST':
        my_model = mymodel.XGBOOST(X_train, y_train, _learning_rate=args.xgboost_learning_rate, _max_depth=args.xgboost_max_depth)
    elif args.arch == 'SVM':
        my_model = mymodel.SVM(X_train, y_train, _kernel=args.svc_kernel, _C=args.svc_C, _random_state=args.svc_random_state)

    #评估模型
    if args.arch == 'XGBOOST':
        myutils.acc_score4xgb(X_train, X_test, y_train, y_test, my_model)
    else:
        myutils.acc_score(X_train, X_test, y_train, y_test, my_model, args.arch)

    #保存模型
    myutils.save_model(my_model, args.save_dir)

if __name__ == '__main__':
    main()