import numpy as np


def load_data(filename):
    data = pd.read_csv(filename)
    return data


def clean_dataframe(data):
    # Creating a copy
    df = data.copy()

    # Check for missing values
    print('**********************')
    print('Missing values:')
    null = df.isnull().sum()
    print(null)

    # Check for unique values
    print('**********************')
    print('Unique values:')
    for col in df.columns:
        print(col, df[col].unique(), "\n")

    # Check the datatypes for each column
    for col in df.columns:
        if df[col].dtype == 'int64':
            print(col, 'has all integers.')
        elif df[col].dtype == 'float64':
            print(col, 'has all floats.')
        else:
            print('**', col, 'has some non-numeric values')

    # Check the index in the non-numeric columns which contain '?'
    print('**********************')
    print('Index positions of ? in col "ca"', df[df['ca'] == '?'].index)
    print('\nIndex positions of ? in col "thal"', df[df['thal'] == '?'].index)

    # Replacing the index positions above
    df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
    df['thal'] = pd.to_numeric(df['thal'], errors='coerce')

    # impute missing values using mean imputation
    mean_ca = np.mean(df['ca'])
    mean_thal = np.mean(df['thal'])

    df['ca'].fillna(mean_ca, inplace=True)
    df['thal'].fillna(mean_thal, inplace=True)

    # check for any remaining missing values
    print('**********************')
    print(df.isnull().sum())
    print(df.columns)
    df['num'] = df['num'].astype('category')
    print(df.info)
    return df


def data_analysis(df):
    # Summary measures
    print(df.describe())

    # Studying the correlation
    corr = df.corr()
    plt.figure(figsize=(15, 15))
    sns.heatmap(corr, annot=True)
    plt.show()

    # Plotting Histograms
    df.hist()
    plt.show()

    # Pair plot for summary
    sns.pairplot(df, hue="num", diag_kind="hist")


def manual_feature_selection(df):
    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Select the features with a high correlation with the target variable
    relevant_features = corr_matrix.index[abs(corr_matrix['num']) > 0.2]

    # Create a new DataFrame with the selected features
    selected_df = df[relevant_features]
    print(selected_df)

    print("Selected columns with manual feature selection: \n ", selected_df.columns)

    # Split the data into feature and target variables
    X = selected_df.drop('num', axis=1)
    y = selected_df['num']

    return X, y


def decision_tree_manual(X, y, max_depth=3, test_size=0.2, random_state=42):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report

    # split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # create and train the decision tree classifier
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    dt.fit(X_train, y_train)

    # Prediction
    y_train_pred = dt.predict(X_train)
    y_test_pred = dt.predict(X_test)

    # print classification report for training set
    print("Classification Report for Training Set in Decision tree:")
    print(classification_report(y_train, y_train_pred, zero_division=0))

    # print classification report for test set
    print("Classification Report for Test Set in Decision tree:")
    print(classification_report(y_test, y_test_pred, zero_division=0))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # group into a binary confusion matrix
    y_train_binary = [1 if x > 1 else 0 for x in y_train]
    y_test_binary = [1 if x > 1 else 0 for x in y_test]
    y_train_pred_binary = [1 if x > 1 else 0 for x in y_train_pred]
    y_test_pred_binary = [1 if x > 1 else 0 for x in y_test_pred]

    # print confusion matrix heatmap for training data
    train_cm = confusion_matrix(y_train_binary, y_train_pred_binary)
    sns.heatmap(train_cm, annot=True, cmap='Blues', ax=axs[0])
    axs[0].set_title('Training Data manual feature DT')
    axs[0].set_xlabel('Predicted Label')
    axs[0].set_ylabel('True Label')

    # print confusion matrix heatmap for test data
    test_cm = confusion_matrix(y_test_binary, y_test_pred_binary)
    sns.heatmap(test_cm, annot=True, cmap='RdPu', ax=axs[1])
    axs[1].set_title('Test Data manual feature DT')
    axs[1].set_xlabel('Predicted Label')
    axs[1].set_ylabel('True Label')

    plt.show()


def random_forest_manual_feature_selection(X, y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report

    # split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # create and train the random forest classifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    rf.fit(X_train, y_train)

    # Prediction
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    # print classification report for training set
    print("Classification Report for Training Set in Random forest:")
    print(classification_report(y_train, y_train_pred, zero_division=0))

    # print classification report for test set
    print("Classification Report for Test Set in Random forest:")
    print(classification_report(y_test, y_test_pred, zero_division=0))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # group into a binary confusion matrix
    y_train_binary = [1 if x > 1 else 0 for x in y_train]
    y_test_binary = [1 if x > 1 else 0 for x in y_test]
    y_train_pred_binary = [1 if x > 1 else 0 for x in y_train_pred]
    y_test_pred_binary = [1 if x > 1 else 0 for x in y_test_pred]

    # print confusion matrix heatmap for training data
    train_cm = confusion_matrix(y_train_binary, y_train_pred_binary)
    sns.heatmap(train_cm, annot=True, cmap="Pastel2", ax=axs[0])
    axs[0].set_title('Training Data manual feature RF')
    axs[0].set_xlabel('Predicted Label')
    axs[0].set_ylabel('True Label')

    # print confusion matrix heatmap for test data
    test_cm = confusion_matrix(y_test_binary, y_test_pred_binary)
    sns.heatmap(test_cm, annot=True, cmap="Pastel1", ax=axs[1])
    axs[1].set_title('Test Data manual feature RF')
    axs[1].set_xlabel('Predicted Label')
    axs[1].set_ylabel('True Label')

    plt.show()

    return rf


from sklearn.linear_model import LinearRegression


def feature_selection_rfe(df):  # automatic feature selection
    # separate the target variable from the feature variables
    X = df.drop('num', axis=1)
    y = df['num']

    # create a linear regression object
    lr = LinearRegression()

    # create the RFE model and select 10 attributes
    rfe = RFE(lr, n_features_to_select=10)
    rfe = rfe.fit(X, y)

    # get the names of the selected features
    selected_features = X.columns[rfe.support_]
    feature_rankings = pd.Series(rfe.ranking_, index=X.columns)

    # print the selected features and their rankings
    print("Selected features:")
    print(selected_features)
    print("\nFeature rankings:")
    print(feature_rankings)

    # create a new DataFrame with the selected features
    dfa = X[selected_features]
    return dfa


from sklearn.tree import DecisionTreeClassifier


def decision_tree_rfe(dfa, y):
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(dfa, y, test_size=0.2, random_state=42)

    # create a decision tree classifier with limited max depth
    dt = DecisionTreeClassifier(max_depth=2, min_samples_split=2, max_leaf_nodes=3)

    # train the classifier on the training data
    dt.fit(X_train, y_train)

    # make predictions on the testing data
    y_train_pred = dt.predict(X_train)
    y_test_pred = dt.predict(X_test)

    # print classification report for training set
    print('AUTOMATIC FEATURE SELECTION')
    print("Classification Report for Training Set in Decision tree:")
    print(classification_report(y_train, y_train_pred, zero_division=0))

    # print classification report for test set
    print("Classification Report for Test Set in Decision tree:")
    print(classification_report(y_test, y_test_pred, zero_division=0))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # group into a binary confusion matrix
    y_train_binary = [1 if x > 2 else 0 for x in y_train]
    y_test_binary = [1 if x > 2 else 0 for x in y_test]
    y_train_pred_binary = [1 if x > 2 else 0 for x in y_train_pred]
    y_test_pred_binary = [1 if x > 2 else 0 for x in y_test_pred]

    # print confusion matrix heatmap for training data
    train_cm = confusion_matrix(y_train_binary, y_train_pred_binary)
    sns.heatmap(train_cm, annot=True, cmap="Paired", ax=axs[0])
    axs[0].set_title('Training Data autofeature DT')
    axs[0].set_xlabel('Predicted Label')
    axs[0].set_ylabel('True Label')

    # print confusion matrix heatmap for test data
    test_cm = confusion_matrix(y_test_binary, y_test_pred_binary)
    sns.heatmap(test_cm, annot=True, cmap="rainbow", ax=axs[1])
    axs[1].set_title('Test Data autofeature DT')
    axs[1].set_xlabel('Predicted Label')
    axs[1].set_ylabel('True Label')

    plt.show()


from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def random_forest_rfe(df, target_col, n_features=10, test_size=0.2, n_estimators=100, max_depth=3, random_state=42):
    # separate the target variable from the feature variables
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # create a random forest classifier object
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    # create the RFE model and select n_features attributes
    rfe = RFE(rf, n_features_to_select=n_features)
    rfe = rfe.fit(X, y)

    # get the names of the selected features
    selected_features = X.columns[rfe.support_]

    # create a new DataFrame with the selected features
    dfa = X[selected_features]

    # split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(dfa, y, test_size=test_size, random_state=random_state)

    # train the random forest classifier on the training data
    rf.fit(X_train, y_train)

    # make predictions on the training and test data
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    # print classification report for training set
    print('AUTOMATIC FEATURE SELECTION')
    print("Classification Report for Training Set in Random forest:")
    print(classification_report(y_train, y_train_pred, zero_division=0))

    # print classification report for test set
    print("Classification Report for Test Set in Random forest:")
    print(classification_report(y_test, y_test_pred, zero_division=0))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # group into a binary confusion matrix
    y_train_binary = [1 if x > 1 else 0 for x in y_train]
    y_test_binary = [1 if x > 1 else 0 for x in y_test]
    y_train_pred_binary = [1 if x > 1 else 0 for x in y_train_pred]
    y_test_pred_binary = [1 if x > 1 else 0 for x in y_test_pred]

    # print confusion matrix heatmap for training data
    train_cm = confusion_matrix(y_train_binary, y_train_pred_binary)
    sns.heatmap(train_cm, annot=True, cmap="Greens", ax=axs[0])
    axs[0].set_title('Training Data autofeature RF')
    axs[0].set_xlabel('Predicted Label')
    axs[0].set_ylabel('True Label')

    # print confusion matrix heatmap for test data
    test_cm = confusion_matrix(y_test_binary, y_test_pred_binary)
    sns.heatmap(test_cm, annot=True, cmap="YlOrRd", ax=axs[1])
    axs[1].set_title('Test Data autofeature RF')
    axs[1].set_xlabel('Predicted Label')
    axs[1].set_ylabel('True Label')

    plt.show()


def main():
    # load data
    filename = "C:/Users/user/Downloads/processed_cleveland.csv"
    df = load_data(filename)

    # clean data
    df_cleaned = clean_dataframe(df)

    # analyze data
    data_analysis(df_cleaned)

    # manual feature selection
    X, y = manual_feature_selection(df_cleaned)

    # decision tree with manual feature selection
    decision_tree_manual(X, y, max_depth=3, test_size=0.2, random_state=42)

    # random forest with manual feature selection
    random_forest_manual_feature_selection(X, y)

    # feature selection with RFE
    feature_selection_rfe(df_cleaned)

    # decision tree with RFE feature selection
    decision_tree_rfe(df_cleaned, df_cleaned['num'])

    # random forest with RFE feature selection
    random_forest_rfe(df_cleaned, 'num')


if __name__ == '__main__':
    main()
