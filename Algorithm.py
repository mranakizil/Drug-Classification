
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from Dataset import *
from Algorithm import *


class Algorithm:
    def __init__(self, X_train, X_test, y_train, y_test):
        
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        # accuracies of each algorithm
        self.knn_acc = 0
        self.nb_acc = 0
    

    def knn(self):
        KNclassifier = KNeighborsClassifier(n_neighbors=20)
        KNclassifier.fit(self.X_train, self.y_train)

        y_pred = KNclassifier.predict(self.X_test)

        print(classification_report(self.y_test, y_pred))
        print(confusion_matrix(self.y_test, y_pred))
        
        self.knn_acc = accuracy_score(y_pred, self.y_test)
        print('K Neighbours accuracy is: {:.2f}%'.format(self.knn_acc*100))
     

    def naive_bayes(self):
        NBclassifier = CategoricalNB()
        NBclassifier.fit(self.X_train, self.y_train)
        y_pred = NBclassifier.predict(self.X_test)

        print(classification_report(self.y_test, y_pred))
        print(confusion_matrix(self.y_test, y_pred))

        self.nb_acc = accuracy_score(y_pred, self.y_test)
        print('Naive Bayes accuracy is: {:.2f}%'.format(self.nb_acc*100))
       
  