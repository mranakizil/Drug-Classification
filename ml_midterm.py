import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


class DrugClassification:
  def __init__(self, df_drug):
    self.df_drug = df_drug


def explore_categorical_variables(self):
    print(self.df_drug.Drug.value_counts())
    print( self.df_drug.Sex.value_counts())
    print(self.df_drug.BP.value_counts())
    print(self.df_drug.Cholesterol.value_counts())
    
def explore_numerical_variables(self):
    skewAge = self.df_drug.Age.skew(axis = 0, skipna = True)
    print('Age skewness: ', skewAge)
    sns.histplot(self.df_drug['Age'], kde=True, stat="density")
    plt.show()

    skewNatoK = self.df_drug.Na_to_K.skew(axis = 0, skipna = True)
    print('Na to K skewness: ', skewNatoK)
    sns.histplot(self.df_drug['Na_to_K'], kde=True, stat="density")
    plt.show()

def exploratory_data_analysis(self):
    # drug type distribution
    sns.set_theme(style="darkgrid")
    sns.countplot(y = "Drug", data = self.df_drug, palette = "flare")
    plt.ylabel('Drug Type')
    plt.xlabel('Total')
    plt.show()

    # gender distribution
    sns.set_theme(style="darkgrid")
    sns.countplot(x = "Sex", data = self.df_drug, palette = "rocket")
    plt.xlabel('Gender (F=Female, M=Male)')
    plt.ylabel('Total')
    plt.show()

    # blood pressure distribution
    sns.set_theme(style="darkgrid")
    sns.countplot(y="BP", data = self.df_drug, palette="crest")
    plt.ylabel('Blood Pressure')
    plt.xlabel('Total')
    plt.show()

    # cholesterol distribution
    sns.set_theme(style="darkgrid")
    sns.countplot(x = "Cholesterol", data = self.df_drug, palette = "magma")
    plt.xlabel('Blood Pressure')
    plt.ylabel('Total')
    plt.show()

    # gender distribution based on drug type
    pd.crosstab(self.df_drug.Sex, self.df_drug.Drug).plot(kind="bar",figsize=(12,5),color=['#003f5c','#ffa600','#58508d','#bc5090','#ff6361'])
    plt.title('Gender distribution based on Drug type')
    plt.xlabel('Gender')
    plt.xticks(rotation = 0)
    plt.ylabel('Frequency')
    plt.show()

    # blood pressure distribution based on cholesterol
    pd.crosstab(self.df_drug.BP, self.df_drug.Cholesterol).plot(kind="bar",figsize=(15,6),color=['#6929c4','#1192e8'])
    plt.title('Blood Pressure distribution based on Cholesterol')
    plt.xlabel('Blood Pressure')
    plt.xticks(rotation=0)
    plt.ylabel('Frequency')
    plt.show()

    # sodium to potassium distribution based on gender and age
    plt.scatter(x = self.df_drug.Age[self.df_drug.Sex=='F'], y = self.df_drug.Na_to_K[(self.df_drug.Sex=='F')], c="Blue")
    plt.scatter(x = self.df_drug.Age[self.df_drug.Sex=='M'], y = self.df_drug.Na_to_K[(self.df_drug.Sex=='M')], c="Orange")
    plt.legend(["Female", "Male"])
    plt.xlabel("Age")
    plt.ylabel("Na_to_K")
    plt.show()

def data_bining(self):
    # divided age into 7 categories
    bin_age = [0, 19, 29, 39, 49, 59, 69, 80]
    category_age = ['<20s', '20s', '30s', '40s', '50s', '60s', '>60s']
    self.df_drug['Age_binned'] = pd.cut(self.df_drug['Age'], bins=bin_age, labels=category_age)
    self.df_drug = self.df_drug.drop(['Age'], axis = 1)

    # divede chemical ratio into 4 categories
    bin_NatoK = [0, 9, 19, 29, 50]
    category_NatoK = ['<10', '10-20', '20-30', '>30']
    self.df_drug['Na_to_K_binned'] = pd.cut(self.df_drug['Na_to_K'], bins=bin_NatoK, labels=category_NatoK)
    self.df_drug = self.df_drug.drop(['Na_to_K'], axis = 1)

def split_dataset(self):
    X = self.df_drug.drop(["Drug"], axis=1)
    y = self.df_drug["Drug"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    return X_train, X_test, y_train, y_test

def feature_engineering(X_train, X_test):
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    return X_train, X_test

def random_over_sampler(X_train, y_train):
    # Randomly over sample the minority class
    ros = RandomOverSampler(random_state=42)
    X_train, y_train= ros.fit_resample(X_train, y_train)
    return X_train, y_train

def smote(X_train, y_train):
    X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    return X_train, y_train

def check_number_of_methods(self, y_train):
    # Check the number of records after over sampling
    print(sorted(Counter(y_train).items()))
    sns.set_theme(style="darkgrid")
    sns.countplot(y=y_train, data=self.df_drug, palette="mako_r")
    plt.ylabel('Drug Type')
    plt.xlabel('Total')
    plt.show()


def knn(X_train, X_test, y_train, y_test):
    KNclassifier = KNeighborsClassifier(n_neighbors=20)
    KNclassifier.fit(X_train, y_train)

    y_pred = KNclassifier.predict(X_test)

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    
    KNAcc = accuracy_score(y_pred,y_test)
    print('K Neighbours accuracy is: {:.2f}%'.format(KNAcc*100))


def categorical_naive_bayes(X_train, X_test, y_train, y_test):
    NBclassifier1 = CategoricalNB()
    NBclassifier1.fit(X_train, y_train)
    y_pred = NBclassifier1.predict(X_test)

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    NBAcc1 = accuracy_score(y_pred,y_test)
    print('Naive Bayes accuracy is: {:.2f}%'.format(NBAcc1*100))

def gaussian_naive_bayes(X_train, X_test, y_train, y_test):
    NBclassifier2 = GaussianNB()
    NBclassifier2.fit(X_train, y_train)

    y_pred = NBclassifier2.predict(X_test)

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    from sklearn.metrics import accuracy_score
    NBAcc2 = accuracy_score(y_pred,y_test)
    print('Gaussian Naive Bayes accuracy is: {:.2f}%'.format(NBAcc2*100))

    
def main():
    df_drug = pd.read_csv("drug200.csv")
    dc = DrugClassification(df_drug)
    dc.df_drug.head()
    dc.df_drug.head()
    # check null in dataset
    print(dc.df_drug.info())

    explore_categorical_variables(dc)
    explore_numerical_variables(dc)
    exploratory_data_analysis(dc)
    data_bining(dc)
    X_train, X_test, y_train, y_test = split_dataset(dc)
    X_train, X_test = feature_engineering(X_train, X_test)
    # X_train, y_train = random_over_sampler(X_train, y_train)
    X_train, y_train = smote(X_train, y_train)
    check_number_of_methods(dc, y_train)
    
    print("---------------------k-NN---------------------")
    knn(X_train, X_test, y_train, y_test)
    print("---------------------categorical naive bayes---------------------")
    categorical_naive_bayes(X_train, X_test, y_train, y_test)
    print("---------------------gaussian naive bayes---------------------")
    gaussian_naive_bayes(X_train, X_test, y_train, y_test)

    

if __name__ == "__main__":
    main()

