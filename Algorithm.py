"""
CSE4088 Introduction to Machine Learning - Project Part #2
Merve Rana Kızıl - 150119825
Elif Gülay - 150119732
Sueda Bilen - 150117044
"""

from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from Dataset import *
from Algorithm import *


class Algorithm:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        # accuracies of each algorithm
        self.knn_acc = 0
        self.nb_acc = 0
        self.rf_acc = 0
        self.svm_acc = 0
        self.dt_acc = 0
    

    def knn(self):
        KNclassifier = KNeighborsClassifier(n_neighbors=20)
        KNclassifier.fit(self.X_train, self.y_train)
        y_pred = KNclassifier.predict(self.X_test)
        print()
        print(classification_report(self.y_test, y_pred))
        self.knn_acc = accuracy_score(y_pred, self.y_test)
        print('K Neighbours accuracy is: {:.2f}%'.format(self.knn_acc*100))
     

    def naive_bayes(self):
        NBclassifier = CategoricalNB()
        NBclassifier.fit(self.X_train, self.y_train)
        y_pred = NBclassifier.predict(self.X_test)
        print()
        print(classification_report(self.y_test, y_pred))
        self.nb_acc = accuracy_score(y_pred, self.y_test)
        print('Naive Bayes accuracy is: {:.2f}%'.format(self.nb_acc*100))
    

    def random_forest(self):
        RFclassifier = RandomForestClassifier(max_leaf_nodes=30)
        RFclassifier.fit(self.X_train, self.y_train)

        y_pred = RFclassifier.predict(self.X_test)

        print(classification_report(self.y_test, y_pred))
        self.rf_acc = accuracy_score(y_pred, self.y_test)
        print('Random Forest accuracy is: {:.2f}%'.format(self.rf_acc*100))
  

    def SVM(self):
        SVCclassifier = SVC(kernel='linear', max_iter=251)
        SVCclassifier.fit(self.X_train, self.y_train)
        y_pred = SVCclassifier.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
        self.svm_acc = accuracy_score(y_pred, self.y_test)
        print('SVM accuracy is: {:.2f}%'.format(self.svm_acc*100))
  

    def decision_tree(self):
        DTclassifier = DecisionTreeClassifier(max_leaf_nodes=20)
        DTclassifier.fit(self.X_train, self.y_train)
        y_pred = DTclassifier.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
        self.dt_acc = accuracy_score(y_pred, self.y_test)
        print('Decision Tree accuracy is: {:.2f}%'.format(self.dt_acc*100))

    def algorithm_comparison(self):
        compare = pd.DataFrame({'Model': ['K Neighbors', 'NB', 'Random Forest', 'SVM', 'Decision Tree'], 'Accuracy': 
        [self.knn_acc*100, self.nb_acc*100, self.rf_acc*100, self.svm_acc*100, self.dt_acc*100]})
        compare.sort_values(by='Accuracy', ascending=False)
        print(compare)
