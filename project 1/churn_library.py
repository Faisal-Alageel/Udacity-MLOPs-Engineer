# library doc string
"""
Helper functions for Predicting customer Churn notebook
author: Faisal Alageel
Date: Sep. 2023
"""

# import libraries
import seaborn as sns
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay
from warnings import filterwarnings
from sklearn.exceptions import DataConversionWarning
filterwarnings(action='ignore')
sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    Returns a pandas dataframe for the CSV file located at the given path.

    Parameters:
        pth (str): The file path to the CSV.

    Returns:
        dataframe (pd.DataFrame): The loaded dataframe.
    '''
    dataframe = pd.read_csv(pth, index_col=0)
    dataframe.dropna(how='all', inplace=True)

    # Encode Churn dependent variable : 0 = Did not churned ; 1 = Churned
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Drop redundant Attrition_Flag variable (replaced by Churn response
    # variable)
    dataframe.drop('Attrition_Flag', axis=1, inplace=True)

    # Drop variables not relevant for the prediction model
    dataframe.drop('CLIENTNUM', axis=1, inplace=True)

    return dataframe


def perform_eda(dataframe):
    '''
    Perform Exploratory Data Analysis on a given dataframe and save figures to the images folder.

    Args:
        dataframe (pd.DataFrame): The pandas dataframe to be analyzed.

    Returns:
        None
    '''

    # Analyze categorical features and plot distribution
    cat_columns = dataframe.select_dtypes(
        include=['object', 'category']).columns.tolist()
    for cat_column in cat_columns:
        plt.figure(figsize=(7, 4))
        (dataframe[cat_column]
            .value_counts()
            .plot(kind='bar', rot=45, title=f'Distribution of {cat_column}')
         )
        plt.savefig(
            os.path.join(
                "./images/eda",
                f'{cat_column}.png'),
            bbox_inches='tight')
        plt.show()
        plt.close()

    num_columns = dataframe.select_dtypes(include='number').columns.tolist()
    for num_column in num_columns:
        plt.figure(figsize=(7, 4))
        (dataframe[num_column]
            .plot(kind='hist', title=f'Distribution of {num_column}')
         )
        plt.savefig(
            os.path.join(
                "./images/eda",
                f'{num_column}.png'),
            bbox_inches='tight')
        plt.show()
        plt.close()

    # Show distribution of 'Total_Trans_Ct' with a kernel density estimate
    plt.figure(figsize=(10, 5))
    sns.histplot(dataframe['Total_Trans_Ct'], stat='density', kde=True)
    plt.show()
    plt.close()

    # Plot correlation matrix
    plt.figure(figsize=(20, 10))
    sns.heatmap(dataframe.corr(), square=True, cmap='Blues', linewidths=0.5)
    plt.title('Correlation Matrix', fontsize=16)
    plt.savefig(
        os.path.join(
            "./images/eda",
            'correlation_matrix.png'),
        bbox_inches='tight')
    plt.show()
    plt.close()


def encoder_helper(dataframe, category_lst, response='Churn'):
    '''
    Helper function to create new columns for each category in categorical columns.

    Args:
        dataframe (pd.DataFrame): Input pandas dataframe.
        category_lst (list): List of columns containing categorical features.
        response (str, optional): Name of the response variable. Defaults to 'Churn'.

    Returns:
        pd.DataFrame: Transformed pandas dataframe.
    '''
    category_groups = {}
    for category in category_lst:
        category_groups[category] = dataframe.groupby(category).mean()[response]
        new_feature = f"{category}_{response}"
        dataframe[new_feature] = dataframe[category].apply(
            lambda x: category_groups[category].loc[x])

    # Drop the obsolete categorical features of the category_lst
    dataframe.drop(category_lst, axis=1, inplace=True)

    return dataframe



def perform_feature_engineering(dataframe, response='Churn'):
    '''
    Encoding categorical features and splitting data into train and test sets.

    Args:
        dataframe (pd.DataFrame): Input pandas dataframe.
        response (str, optional): Name of the response variable. Defaults to 'Churn'.

    Returns:
        x_train (pd.DataFrame): Training data features.
        x_test (pd.DataFrame): Testing data features.
        y_train (pd.Series): Training data response variable.
        y_test (pd.Series): Testing data response variable.
    '''

    # Collect categorical features to be encoded
    cat_columns = dataframe.select_dtypes(include='object').columns.tolist()

    # Encode categorical features using mean of response variable on category
    dataframe = encoder_helper(dataframe, cat_columns, response='Churn')
    features = dataframe.drop(response, axis=1)
    target = dataframe[response]

    # train test split
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.3, random_state=42)

    return features_train, features_test, target_train, target_test





def plot_classification_report(
        model_name,
        target_train,
        target_test,
        target_train_preds,
        target_test_preds):
    '''
    produces classification report for training and testing results and stores
    report as image in images folder

    input:
                    model_name: (str) name of the model, ie 'Random Forest'
                    y_train: training response values
                    y_test:  test response values
                    y_train_preds: training predictions from model_name
                    y_test_preds: test predictions from model_name

    output:
                     None
    '''

    plt.rc('figure', figsize=(5, 5))

    # Plot Classification report on Train dataset
    plt.text(0.01, 1.25,
             str(f'{model_name} Train'),
             {'fontsize': 10},
             fontproperties='monospace'
             )
    plt.text(0.01, 0.05,
             str(classification_report(target_train, target_train_preds)),
             {'fontsize': 10},
             fontproperties='monospace'
             )

    # Plot Classification report on Test dataset
    plt.text(0.01, 0.6,
             str(f'{model_name} Test'),
             {'fontsize': 10},
             fontproperties='monospace'
             )
    plt.text(0.01, 0.7,
             str(classification_report(target_test, target_test_preds)),
             {'fontsize': 10},
             fontproperties='monospace'
             )

    plt.axis('off')

    # Save figure to ./images folder
    fig_name = f'Classification_report_{model_name}.png'
    plt.savefig(
        os.path.join(
            "./images/results",
            fig_name),
        bbox_inches='tight')

    # Display figure
    plt.show()
    plt.close()


def classification_report_image(
        target_train,
        target_test,
        target_train_preds_lr,
        target_train_preds_rf,
        target_test_preds_lr,
        target_test_preds_rf):
    '''
    Produces classification reports for training and testing results and stores them as images ,
    using the plot_classification_report helper function.

    Args:
        y_train (array-like): True training response values.
        y_test (array-like): True test response values.
        y_train_preds_lr (array-like): Training predictions from logistic regression.
        y_train_preds_rf (array-like): Training predictions from random forest.
        y_test_preds_lr (array-like): Test predictions from logistic regression.
        y_test_preds_rf (array-like): Test predictions from random forest.

    Returns:
        None
    '''

    plot_classification_report('Logistic Regression',
                               target_train,
                               target_test,
                               target_train_preds_lr,
                               target_test_preds_lr)

    plot_classification_report('Random Forest',
                               target_train,
                               target_test,
                               target_train_preds_rf,
                               target_test_preds_rf)


def feature_importance_plot(model, X_data, model_name, output_pth):
    '''
    Creates and stores the feature importances in the specified path.

    Args:
        model : object
            Model object containing feature_importances_.
        X_data : pd.DataFrame
            Pandas dataframe of X values.
        model_name : str
            Name of the model.
        output_pth : str
            Path to store the figure.

    Returns:
        None
    '''

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title(f"Feature Importance for {model_name}")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save figure to output_pth
    fig_name = f'feature_importance_{model_name}.png'
    plt.savefig(os.path.join(output_pth, fig_name), bbox_inches='tight')

    # Display feature importance figure
    plt.show()
    plt.close()


def confusion_matrix(model, model_name, X_test, y_test):
    '''
    Display the confusion matrix of a model on test data.

    Args:
        model: Trained model.
        model_name: Name of the model.
        X_test: X testing data.
        y_test: y testing data.

    Returns:
        None
    '''

    class_names = ['Not Churned', 'Churned']

    plt.figure(figsize=(15, 5))
    ax = plt.gca()

    y_pred = model.predict(X_test)
    cm_display = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        xticks_rotation='horizontal',
        ax=ax)
    cm_display.ax_.grid(False)
    plt.title(f'{model_name} Confusion Matrix on Test Data')
    plt.savefig(
        os.path.join(
            "./images/results",
            f'{model_name}_Confusion_Matrix.png'), bbox_inches='tight')
    plt.show()
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    Train models, store results (images + scores), and store models.

    Args:
        X_train (pd.DataFrame): X training data.
        X_test (pd.DataFrame): X testing data.
        y_train (pd.Series): y training data.
        y_test (pd.Series): y testing data.

    Returns:
        None
    '''
    # Initialize Random Forest model
    rfc = RandomForestClassifier(random_state=42)

    # Initialize Logistic Regression model
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

    # Grid search for random forest parameters and instantiation
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    # Train Random Forest using GridSearch
    cv_rfc.fit(X_train, y_train)

    # Train Logistic Regression
    lrc.fit(X_train, y_train)

    # Save best models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Get predictions
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Calculate classification scores
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # Plot ROC-curves
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    RocCurveDisplay.from_estimator(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8
    )

    RocCurveDisplay.from_estimator(lrc, X_test, y_test, ax=ax, alpha=0.8)

    # Save ROC-curves to images directory
    plt.savefig(
        os.path.join(
            "./images/results",
            'ROC_curves.png'), bbox_inches='tight')
    plt.show()
    plt.close()

    for model, model_type in zip([cv_rfc.best_estimator_, lrc],
                                 ['Random_Forest', 'Logistic_Regression']
                                 ):
        # Display confusion matrix on test data
        confusion_matrix(model, model_type, X_test, y_test)

    # Display feature importance on train data
    feature_importance_plot(cv_rfc.best_estimator_,
                            X_train,
                            'Random_Forest',
                            "./images/results")

    fig_name = 'shap_values_random_forest.png'
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    plt.savefig(
        os.path.join(
            "./images/results",
            fig_name),
        bbox_inches='tight')
    plt.show()
    plt.close()

    # Summary plot for Class = 1 (churned)
    shap.summary_plot(shap_values[1], X_test, show=False)
    plt.savefig('./images/results/summary_shap_values_random_forest.png')
    plt.show()
    plt.close()


if __name__ == "__main__":
    dataset = import_data("./data/bank_data.csv")
    print('Dataset successfully loaded...Now conducting data exploration')
    perform_eda(dataset)
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        dataset, response='Churn')
    print('Start training the model...please wait')
    train_models(X_train, X_test, y_train, y_test)
    print('Training completed. Best model weights + performance matrics saved')
