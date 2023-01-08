"""
CSE4088 Introduction to Machine Learning - Project Part #2
Merve Rana Kızıl - 150119825
Elif Gülay - 150119732
Sueda Bilen - 150117044
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter


class Dataset:
    def __init__(self, df_drug):
        self.df_drug = df_drug

    def data_bining(self):
        # divided age into 7 categories
        bin_age = [0, 19, 29, 39, 49, 59, 69, 80]
        category_age = ['<20s', '20s', '30s', '40s', '50s', '60s', '>60s']
        self.df_drug['Age_binned'] = pd.cut(self.df_drug['Age'], bins=bin_age, labels=category_age)
        self.df_drug = self.df_drug.drop(['Age'], axis = 1)

        # divide chemical ratio into 4 categories
        bin_NatoK = [0, 9, 19, 29, 50]
        category_NatoK = ['<10', '10-20', '20-30', '>30']
        self.df_drug['Na_to_K_binned'] = pd.cut(self.df_drug['Na_to_K'], bins=bin_NatoK, labels=category_NatoK)
        self.df_drug = self.df_drug.drop(['Na_to_K'], axis = 1)


    def explore_categorical_variables(self):
        print(self.df_drug.Drug.value_counts())
        print(self.df_drug.Sex.value_counts())
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
        sns.countplot(y = "Drug", data = self.df_drug, palette = "icefire")
        plt.ylabel('Drug Type')
        plt.xlabel('Total')
        plt.show()

        # gender distribution
        sns.set_theme(style="darkgrid")
        sns.countplot(x = "Sex", data = self.df_drug, palette = "coolwarm")
        plt.xlabel('Gender (F=Female, M=Male)')
        plt.ylabel('Total')
        plt.show()

        # blood pressure distribution
        sns.set_theme(style="darkgrid")
        sns.countplot(y="BP", data = self.df_drug, palette="vlag")
        plt.ylabel('Blood Pressure')
        plt.xlabel('Total')
        plt.show()

        # cholesterol distribution
        sns.set_theme(style="darkgrid")
        sns.countplot(x = "Cholesterol", data = self.df_drug, palette = "crest")
        plt.xlabel('Blood Pressure')
        plt.ylabel('Total')
        plt.show()

        # gender distribution based on drug type
        pd.crosstab(self.df_drug.Sex, self.df_drug.Drug).plot(kind="bar",figsize=(12,5), color=['#9BC2B2','#C5D6BA','#F2E9D3','#F6C8B6','#CA9CAC'])
        plt.title('Gender distribution based on Drug type')
        plt.xlabel('Gender')
        plt.xticks(rotation = 0)
        plt.ylabel('Frequency')
        plt.show()

        # blood pressure distribution based on cholesterol
        pd.crosstab(self.df_drug.BP, self.df_drug.Cholesterol).plot(kind="bar",figsize=(15,6), color=['#FFCBB5','#F28589'])
        plt.title('Blood Pressure distribution based on Cholesterol')
        plt.xlabel('Blood Pressure')
        plt.xticks(rotation=0)
        plt.ylabel('Frequency')
        plt.show()

        # sodium to potassium distribution based on gender and age
        plt.scatter(x = self.df_drug.Age[self.df_drug.Sex=='F'], y = self.df_drug.Na_to_K[(self.df_drug.Sex=='F')], c="Green")
        plt.scatter(x = self.df_drug.Age[self.df_drug.Sex=='M'], y = self.df_drug.Na_to_K[(self.df_drug.Sex=='M')], c="Blue")
        plt.legend(["Female", "Male"])
        plt.xlabel("Age")
        plt.ylabel("Na_to_K")
        plt.show()


    def split_dataset(self):
        X = self.df_drug.drop(["Drug"], axis=1)
        y = self.df_drug["Drug"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
        return X_train, X_test, y_train, y_test


    def feature_engineering(self, X_train, X_test):
        X_train = pd.get_dummies(X_train)
        X_test = pd.get_dummies(X_test)
        return X_train, X_test


    def smote(self, X_train, y_train):
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)
        return X_train, y_train


    def check_number_of_methods(self, y_train):
        # Check the number of records after oversampling
        print(sorted(Counter(y_train).items()))
        sns.set_theme(style="darkgrid")
        sns.countplot(y=y_train, data = self.df_drug, palette="cubehelix")
        plt.ylabel('Drug Type')
        plt.xlabel('Total')
        plt.show()