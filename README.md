# Phishing Website Detection
creator: cfg554
### data 
https://archive.ics.uci.edu/dataset/379/website+phishing

### model
我们包含了以下模型
* decision_tree
* random_forest
* MLP
* SVM
* XGBOOST <br>
同时使用了shell脚本文件运行，方便自主选择模型和参数

### 运行步骤(为了防止冲突，请自己创建全新的虚拟环境，python环境为3.9)
step1:
    pip install -r requirements.txt
step2:
    在run.sh所在目录下提权
    chmod +x run.sh
step3:
    ./run.sh

<br>
非常简单易用，欢迎star🥰
（work.ipynb为分解的详细步骤）
    
