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


# $B71N}%G!<%?(B $B!'6v?t9T(B
X_train, Y_train = X[0::2], Y[0::2]
# $B%F%9%H%G!<%?!'4q?t9T(B
X_test, Y_test = X[1::2], Y[1::2]


clf = svm.SVC(gamma=0.001)
# $B71N}%G!<%?$H%i%Y%k$G3X=,(B
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)

print(accuracy)

# $B3X=,:Q%b%G%k$r;H$C$F%F%9%H%G!<%?$rJ,N`$7$?7k2L$rJV$9(B
predicted = clf.predict(X_test)

# $B>\$7$$%l%]!<%H(B
# precision($BE,9gN((B): $BA*Br$7$?@52r(B/$BA*Br$7$?=89g(B
# recall($B:F8=N((B) : $BA*Br$7$?@52r(B/$BA4BN$N@52r(B
# F-score(F$BCM(B) : $BE,9gN($H:F8=N($O%H%l!<%I%*%U$N4X78$K$"$k$?$a(B
print("classification report")
print(metrics.classification_report(Y_test, predicted))

