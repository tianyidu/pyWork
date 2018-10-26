from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    modelname = os.path.join(os.path.dirname(__file__),"decision.joblib")
    data = pd.read_csv(r"F:\work\tmp\m01_user_info_all__201810241510.csv", low_memory=False)
    # print(data["status_str"].unique())
    data = data[data["native_place"] != "Y3"]

    x = data.drop(["usersid", "companyaddrch", "username", "status_str", "native_place_str", "sex_str", "birthday",
                   "usertitle_str", "deptname", "shopname",
                   "enterdate", "leavedate", "bp_bak", "certificate_str", "caldate", "iswebbroker_str", "phs",
                   "iswbrokerleader_str", "is5i5jwebuser_str", "brokercard"], axis=1)

    x = x.fillna(0)
    y = data["status_str"].map({"离职": 0, "在职": 1})
    std = StandardScaler()
    x = std.fit_transform(x)
    print(type(x))
    pca = PCA(n_components=20)
    x = pca.fit_transform(x)
    print(type(x))
    train_x = x[:80000]
    train_y = y[:80000]
    eval_x = x[80000:]
    eval_y = y[80000:]
    print(eval_x.shape)

    if not os.path.exists(modelname):
        print("training...")
        clf = tree.DecisionTreeClassifier()
        clf.fit(train_x, train_y)
        joblib.dump(clf,modelname)
    else:
        print("load model")
        clf = joblib.load(modelname)

    # eval_x = np.array(x[98502]).reshape([1,-1])
    # print(eval_x)
    pre_y = clf.predict(eval_x)
    print(np.mean(np.array(pre_y)==np.array(eval_y)))
    # print("predict value:",pre_y,"eval value",eval_y.loc[98502])


