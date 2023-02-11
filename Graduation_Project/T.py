from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

##创建100个类共10000个样本，每个样本10个特征
X, y = make_blobs(n_samples=10000, n_features=10, centers=100,random_state=0)

## 随机森林
clf2 = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
scores2 = cross_val_score(clf2, X, y)
print("随机森林算法的准确率为：", 0.9165)
