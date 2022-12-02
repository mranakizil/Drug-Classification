"""
CSE4088 Introduction to Machine Learning - Project Part #2
Merve Rana Kızıl - 150119825
Elif Gülay - 150119732
Sueda Bilen - 150117044
"""

import pandas as pd
from Dataset import *
from Algorithm import *


def main():
    df_drug = pd.read_csv("drug200.csv")
    print("Dataset head: ")
    print(df_drug.head())
    # check null in dataset
    print("Dataset info:")
    print(df_drug.info())

    dataset = Dataset(df_drug)
   
    dataset.explore_categorical_variables()
    dataset.explore_numerical_variables()
    dataset.exploratory_data_analysis()
    dataset.data_bining()
    X_train, X_test, y_train, y_test = dataset.split_dataset()
    X_train, X_test = dataset.feature_engineering(X_train, X_test)
    # print(X_train.head())
    # dataset.check_number_of_methods(y_train)
    X_train, y_train = dataset.smote(X_train, y_train)
    dataset.check_number_of_methods(y_train)

    X = df_drug.drop(["Drug"], axis=1)
    y = df_drug["Drug"]
    algorithm = Algorithm(X_train, X_test, y_train, y_test)
    print("-----------------------k-NN-----------------------")
    algorithm.knn()
    print("-----------------------Naive Bayes-----------------------")
    algorithm.naive_bayes()
    

if __name__ == "__main__":
    main()

