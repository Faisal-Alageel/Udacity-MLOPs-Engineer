# library doc string
"""
Helper functions for Predicting customer Churn notebook
author: Faisal Alageel
Date: Sep. 2023
"""

# import libraries
import os
from warnings import filterwarnings

import seaborn as sns
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, plot_confusion_matrix, plot_roc_curve
import pandas as pd
import numpy as np

filterwarnings(action='ignore')
sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

def import_data(pth):
    """
    Load a CSV file and return its data as a pandas DataFrame.

    Parameters:
        pth (str): The file path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame containing the data.
    """
    dataframe = pd.read_csv(pth, index_col=0)
    dataframe.dropna(how='all', inplace=True)

    # Encode Churn dependent variable: 0 = Did not churn; 1 = Churned
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Drop the redundant Attrition_Flag variable (replaced by the Churn response variable)
    dataframe.drop('Attrition_Flag', axis=1, inplace=True)

    # Drop variables not relevant for the prediction model
    dataframe.drop('CLIENTNUM', axis=1, inplace=True)

    return dataframe


def perform_eda(dataframe, show_fig=True):
    """
    Perform EDA on a given DataFrame and save figures to the 'images' folder.

    Args:
        dataframe (pd.DataFrame): The pandas DataFrame to be analyzed.
        show_fig (bool, optional): Whether to display the generated figures. Defaults to True.

    Returns:
        None
    """

    # Analyze categorical features and plot their distributions
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
        if show_fig:
            plt.show()
        plt.close()

    # Analyze numerical features and plot their distributions
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
        if show_fig:
            plt.show()
        plt.close()

    # Show distribution of 'Total_Trans_Ct' with a kernel density estimate
    plt.figure(figsize=(10, 5))
    sns.histplot(dataframe['Total_Trans_Ct'], stat='density', kde=True)
    if show_fig:
        plt.show()
    plt.close()

    # Plot the correlation matrix
    plt.figure(figsize=(20, 10))
    sns.heatmap(dataframe.corr(), square=True, cmap='Blues', linewidths=0.5)
    plt.title('Correlation Matrix', fontsize=16)
    plt.savefig(
        os.path.join(
            "./images/eda",
            'correlation_matrix.png'),
        bbox_inches='tight')
    if show_fig:
        plt.show()
    plt.close()


def encoder_helper(dataframe, category_lst, response='Churn'):
    """
    Encode categorical features in a DataFrame by creating new columns for each category.

    Args:
        dataframe (pd.DataFrame): Input pandas DataFrame.
        category_lst (list): List of columns containing categorical features.
        response (str, optional): Name of the response variable. Defaults to 'Churn'.

    Returns:
        pd.DataFrame: Transformed pandas DataFrame.
    """
    category_groups = {}
    for category in category_lst:
        category_groups[category] = dataframe.groupby(category).mean()[
            response]
        new_feature = f"{category}_{response}"
        dataframe[new_feature] = dataframe[category].apply(
            lambda x: category_groups[category].loc[x])

    # Drop the obsolete categorical features from the category_lst
    dataframe.drop(category_lst, axis=1, inplace=True)

    return dataframe


def perform_feature_engineering(dataframe, response='Churn'):
    """
    Encode categorical features and split data into training and testing sets.

    Args:
        dataframe (pd.DataFrame): Input pandas DataFrame.
        response (str, optional): Name of the response variable. Defaults to 'Churn'.

    Returns:
        x_train (pd.DataFrame): Training data features.
        x_test (pd.DataFrame): Testing data features.
        y_train (pd.Series): Training data response variable.
        y_test (pd.Series): Testing data response variable.
    """

    # Collect categorical features to be encoded
    cat_columns = dataframe.select_dtypes(include='object').columns.tolist()

    # Encode categorical features using the mean of the response variable by category
    dataframe = encoder_helper(dataframe, cat_columns, response='Churn')
    features = dataframe.drop(response, axis=1)
    target = dataframe[response]

    # Train-test split
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.3, random_state=42)

    return features_train, features_test, target_train, target_test

def plot_classification_report(
        model_name,
        target_train,
        target_test,
        target_train_preds,
        target_test_preds,
        show_fig=True):
    """
    Generate a classification report for training and testing results, and save it as png.

    Args:
        model_name (str): Name of the model, e.g., 'Random Forest'.
        target_train (array-like): Training response values.
        target_test (array-like): Test response values.
        target_train_preds (array-like): Training predictions from the model.
        target_test_preds (array-like): Test predictions from the model.
        show_fig (bool, optional): Whether to display the generated figure. Defaults to True.

    Returns:
        None
    """

    plt.rc('figure', figsize=(5, 5))

    # Plot Classification report on Train dataset
    plt.text(0.01, 1.25,
             str(f'{model_name} Train'),
             {'fontsize': 10},
             fontproperties='monospace'
             )
    plt.text(0.01, 0.05,
             str(classification_report(target_test, target_test_preds)),
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
             str(classification_report(target_train, target_train_preds)),
             {'fontsize': 10},
             fontproperties='monospace'
             )

    plt.axis('off')

    # Save the figure to the "./images/results" folder
    fig_name = f'Classification_report_{model_name}.png'
    plt.savefig(
        os.path.join(
            "./images/results",
            fig_name),
        bbox_inches='tight')

    # Display the figure if show_fig is True
    if show_fig:
        plt.show()
    plt.close()


def classification_report_image(
        target_train,
        target_test,
        target_train_preds_lr,
        target_train_preds_rf,
        target_test_preds_lr,
        target_test_preds_rf,
        show_fig=True):
    """
    Generate classification reports and save them as images for training and testing results
    using the plot_classification_report helper function.

    Args:
        target_train (array-like): True training response values.
        target_test (array-like): True test response values.
        target_train_preds_lr (array-like): Training predictions from logistic regression.
        target_train_preds_rf (array-like): Training predictions from random forest.
        target_test_preds_lr (array-like): Test predictions from logistic regression.
        target_test_preds_rf (array-like): Test predictions from random forest.
        show_fig (bool, optional): Whether to display the generated figures. Defaults to True.

    Returns:
        None
    """

    plot_classification_report('Logistic_Regression',
                               target_train,
                               target_test,
                               target_train_preds_lr,
                               target_test_preds_lr,
                               show_fig)

    plot_classification_report('Random_Forest',
                               target_train,
                               target_test,
                               target_train_preds_rf,
                               target_test_preds_rf,
                               show_fig)


def feature_importance_plot(
        model,
        features,
        model_name,
        output_pth,
        show_fig=True):
    """
    Create and store a feature importance plot for a given model.

    Args:
        model : object
            Model object containing feature_importances_.
        features : pd.DataFrame
            Pandas DataFrame of feature data.
        model_name : str
            Name of the model.
        output_pth : str
            Path to store the feature importance plot image.
        show_fig (bool, optional): Whether to display the generated figure. Defaults to True.

    Returns:
        None
    """

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [features.columns[i] for i in indices]

    # Create the plot
    plt.figure(figsize=(20, 5))

    # Create the plot title
    plt.title(f"Feature Importance for {model_name}")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(features.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(features.shape[1]), names, rotation=90)

    # Save the figure to output_pth
    fig_name = f'feature_importance_{model_name}.png'
    plt.savefig(os.path.join(output_pth, fig_name), bbox_inches='tight')

    # Display the feature importance figure if show_fig is True
    if show_fig:
        plt.show()
    plt.close()


def confusion_matrix(
        model,
        model_name,
        features_test,
        target_test,
        show_fig=True):
    """
    Display the confusion matrix of a model on test data.

    Args:
        model: Trained model.
        model_name: Name of the model.
        features_test: Testing data features.
        target_test: True testing data response variable.
        show_fig (bool, optional): Whether to display the generated figure. Defaults to True.

    Returns:
        None
    """

    class_names = ['Not Churned', 'Churned']

    plt.figure(figsize=(15, 5))
    axis = plt.gca()

    # target_pred = model.predict(features_test)
    cm_display = plot_confusion_matrix(
        model,
        features_test,
        target_test,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        xticks_rotation='horizontal',
        ax=axis)
    cm_display.ax_.grid(False)
    plt.title(f'{model_name} Confusion Matrix on Test Data')
    plt.savefig(
        os.path.join(
            "./images/results",
            f'{model_name}_Confusion_Matrix.png'), bbox_inches='tight')
    if show_fig:
        plt.show()
    plt.close()


def train_models(
        features_train,
        features_test,
        target_train,
        target_test,
        show_fig=True):
    """
    Train machine learning models, store results (images + scores), and save models.

    Args:
        features_train (pd.DataFrame): Training data features.
        features_test (pd.DataFrame): Testing data features.
        target_train (pd.Series): Training data response variable.
        target_test (pd.Series): Testing data response variable.
        show_fig (bool, optional): Whether to display the generated figures. Defaults to True.

    Returns:
        None
    """
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
    cv_rfc.fit(features_train, target_train)

    # Train Logistic Regression
    lrc.fit(features_train, target_train)

    # Save best models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Get predictions
    target_train_preds_rf = cv_rfc.best_estimator_.predict(features_train)
    target_test_preds_rf = cv_rfc.best_estimator_.predict(features_test)

    target_train_preds_lr = lrc.predict(features_train)
    target_test_preds_lr = lrc.predict(features_test)

    # Calculate classification scores
    classification_report_image(target_train,
                                target_test,
                                target_train_preds_lr,
                                target_train_preds_rf,
                                target_test_preds_lr,
                                target_test_preds_rf,show_fig)

    # Plot ROC-curves
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        features_test,
        target_test,
        ax=axis,
        alpha=0.8
    )

    plot_roc_curve(lrc, features_test, target_test, ax=axis, alpha=0.8)

    # Save ROC-curves to images directory
    plt.savefig(
        os.path.join(
            "./images/results",
            'ROC_curves.png'), bbox_inches='tight')
    if show_fig:
        plt.show()
    plt.close()

    for model, model_type in zip([cv_rfc.best_estimator_, lrc],
                                 ['Random_Forest', 'Logistic_Regression']
                                 ):
        # Display confusion matrix on test data
        confusion_matrix(model, model_type, features_test, target_test,show_fig)

    # Display feature importance on train data
    feature_importance_plot(cv_rfc.best_estimator_,
                            features_train,
                            'Random_Forest',
                            "./images/results",show_fig)

    fig_name = 'shap_values_random_forest.png'
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(features_test)
    shap.summary_plot(shap_values, features_test, plot_type="bar", show=False)
    plt.savefig(
        os.path.join(
            "./images/results",
            fig_name),bbox_inches='tight')
    if show_fig:
        plt.show()
    plt.close()

    # Summary plot for Class = 1 (churned)
    shap.summary_plot(shap_values[1], features_test, show=False)
    plt.savefig('./images/results/summary_shap_values_random_forest.png',bbox_inches='tight')
    if show_fig:
        plt.show()
    plt.close()


if __name__ == "__main__":
    dataset = import_data("./data/bank_data.csv")
    print('Dataset successfully loaded...Now conducting data exploration')
    perform_eda(dataset, show_fig=False)
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        dataset, response='Churn')
    print('Start training the model...please wait')
    train_models(X_train, X_test, y_train, y_test, show_fig=False)
    print('Training completed. Best model weights + performance matrics saved')
