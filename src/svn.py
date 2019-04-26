# -*- conding:utf-8 -*-

from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt

digits = datasets.load_digits()

# plt.matshow(digits.images[0], cmap="Greys")

# plt.show()

X = digits.data
print(X)

Y = digits.target
print(Y)


# 訓練データ ：偶数行
X_train, Y_train = X[0::2], Y[0::2]
# テストデータ：奇数行
X_test, Y_test = X[1::2], Y[1::2]


clf = svm.SVC(gamma=0.001)
# 訓練データとラベルで学習
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)

print(accuracy)

# 学習済モデルを使ってテストデータを分類した結果を返す
predicted = clf.predict(X_test)

# 詳しいレポート
# precision(適合率): 選択した正解/選択した集合
# recall(再現率) : 選択した正解/全体の正解
# F-score(F値) : 適合率と再現率はトレードオフの関係にあるため
print("classification report")
print(metrics.classification_report(Y_test, predicted))

