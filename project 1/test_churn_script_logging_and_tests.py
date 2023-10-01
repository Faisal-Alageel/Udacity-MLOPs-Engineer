"""
Unit test of churn_library.py module with pytest
author: Faisal Alageel
Date: Sep. 2023
"""

import logging
import pytest
from churn_library import import_data, perform_feature_engineering
from churn_library import encoder_helper, train_models, perform_eda


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


@pytest.fixture(scope="module")
def path():
    """
    Get the path to the bank data CSV file.

    Returns:
        str: The path to the bank data CSV file.
    """
    return "./data/bank_data.csv"


@pytest.fixture(scope="module")
def dataframe(file_path):
    """
    Create a pandas DataFrame from a CSV file located at the given path.

    Args:
        path (str): The file path to the CSV file containing the data.

    Returns:
        pandas.DataFrame: A DataFrame containing the data from the CSV file.

    Example:
        To load data from a CSV file named 'bank_data.csv' in the './data/' directory:
        >>> df = dataframe("./data/bank_data.csv")
    """
    return import_data(file_path)


@pytest.fixture(scope="module",
                params=[['Gender',
                         'Education_Level',
                         'Marital_Status',
                         'Income_Category',
                         'Card_Category'],
                        ['Gender',
                         'Education_Level',
                         'Marital_Status',
                         'Income_Category'],
                        ['Gender',
                         'Education_Level',
                         'Marital_Status',
                         'Income_Category',
                         'Card_Category',
                         'Not_a_column']])
def encoder_params(request):
    """
    Prepare data and categorical feature list for encoder testing.

    Args:
        request: A pytest request object containing parameters.

    Returns:
        tuple: A tuple containing two elements.
            - data (pandas.DataFrame): A copy of the DataFrame used for testing.
            - cat_features (list): A list of categorical feature names for testing.

    Example:
        To prepare data and categorical features for encoder testing:
        >>> data, cat_features = encoder_params(request)
    """
    cat_features = request.param
    data = pytest.df.copy()
    return data, cat_features


@pytest.fixture(scope="module")
def input_train():
    """
    Prepare the training data by performing feature engineering.

    Returns:
        pandas.DataFrame: A DataFrame containing the training data with engineered features.

    Example:
        To prepare training data with feature engineering:
        >>> training_data = input_train()
    """
    data = pytest.df
    return perform_feature_engineering(data)


@pytest.mark.parametrize("filename",
                         ["./data/bank_data.csv",
                          "./data/no_file.csv"])
def test_import(filename):
    """
    Test the import_data function with the given filename.

    This function tests the import_data function by importing data from the filename.
    If the import is successful, it logs information about the success,
    and sets the DataFrame in pytest context.
    If the file is not found, it logs an error message.

    Args:
        filename (str): The name of the file to import.

    Returns:
        None

    Example:
        To test importing data from a file named 'data.csv':
        >>> test_import("data.csv")
    """
    two_test_level = False

    try:
        data = import_data(filename)
        logging.info("Testing import_data from file: %s - SUCCESS", filename)
        pytest.df = data
        two_test_level = True

    except FileNotFoundError:
        logging.error(
            "Testing import_data from file: %s: The file wasn't found",
            filename)

    if two_test_level:
        try:
            assert data.shape[0] > 0
            assert data.shape[1] > 0
            logging.info("Returned dataframe with shape: %s", data.shape)

        except AssertionError:
            logging.error("The file doesn't appear to have rows and columns")


def test_eda():
    """
    Test the perform_eda function with the provided DataFrame.

    This function tests the perform_eda function by calling it with the given DataFrame.
    If the EDA (Exploratory Data Analysis) process is successful,
    it logs information about the success.
    If an error occurs during the EDA process, it logs an error message and error type.

    Args:
        None

    Returns:
        None

    Example:
        To test exploratory data analysis on a DataFrame:
        >>> test_eda()
    """
    data = pytest.df

    try:
        perform_eda(data, show_fig=False)
        logging.info("Testing perform_eda - SUCCESS")

    except Exception as err:
        logging.error("Testing perform_eda failed - Error type %s", type(err))


def test_encoder_helper(encoder_params):
    """
    Test the encoder_helper function with the provided data and categorical features.

    This function tests the encoder_helper function,
    by calling it with the provided data and categorical features.
    If the encoding process is successful, it logs information about the success,
    and checks that all categorical
    columns have been encoded.
    If an error occurs during the encoding process, it logs an error message and the error type.

    Args:
        encoder_parameters (tuple): A tuple containing two elements:
            - data (pandas.DataFrame): The DataFrame containing the data to be encoded.
            - cat_features (list): A list of categorical feature names to be encoded.

    Returns:
        None

    Example:
        To test the encoding of categorical features in a DataFrame:
        >>> test_encoder_helper((data, ["cat_feature_1", "cat_feature_2"]))
    """
    two_test_level = False
    data, cat_features = encoder_params

    try:
        newdf = encoder_helper(data, cat_features)
        logging.info("Testing encoder_helper with %s - SUCCESS", cat_features)
        two_test_level = True

    except KeyError:
        logging.error(
            "Testing encoder_helper with %s failed: Check for categorical features not in dataset",
            cat_features)

    except Exception as err:
        logging.error(
            "Testing encoder_helper failed - Error type %s",
            type(err))

    if two_test_level:
        try:
            assert newdf.select_dtypes(include='object').columns.tolist() == []
            logging.info("All categorical columns were encoded")

        except AssertionError:
            logging.error(
                """At least one categorical column was NOT encoded -
                   Check categorical features submitted""")


def test_perform_feature_engineering():
    """
    Test the perform_feature_engineering function.

    tests the perform_feature_engineering function by calling it with the provided data.
    If the feature engineering process is successful, it logs information about the success
    and checks the shapes of the resulting train and test datasets.
    If an error occurs, it logs an error message and error type.

    Args:
        None

    Returns:
        None

    Example:
        To test feature engineering on a dataset:
        >>> test_perform_feature_engineering()
    """
    two_test_level = False
    try:
        data = pytest.df
        features_train, features_test, target_train, target_test = perform_feature_engineering(
            data)
        logging.info("Testing perform_feature_engineering - SUCCESS")
        two_test_level = True

    except Exception as err:
        logging.error(
            "Testing perform_feature_engineering failed - Error type %s",
            type(err))

    if two_test_level:
        try:
            assert features_train.shape[0] > 0
            assert features_train.shape[1] > 0
            assert features_test.shape[0] > 0
            assert features_test.shape[1] > 0
            assert target_train.shape[0] > 0
            assert target_test.shape[0] > 0
            logging.info(
                "perform_feature_engineering returned Train / Test set of shape %s %s",
                features_train.shape,
                features_test.shape)

        except AssertionError:
            logging.error(
                "The returned train / test datasets do not appear to have rows and columns")


def test_train_models(input_train):
    """
    Test the train_models function with the provided training data.

    tests the train_models function by calling it with the provided training data.
    If the training process is successful, it logs information about the success.
    If an error occurs during the training process, it logs an error message and error type.

    Args:
        input_train (pandas.DataFrame): The training data for model training.

    Returns:
        None

    Example:
        To test training machine learning models with training data:
        >>> test_train_models(training_data)
    """
    try:
        train_models(*input_train, show_fig=False)
        logging.info("Testing train_models: SUCCESS")
    except Exception as err:
        logging.error("Testing train_models failed - Error type %s", type(err))


if __name__ == "__main__":
    test_import("./data/bank_data.csv")
    test_import("./data/no_file.csv")
    test_perform_feature_engineering()
    test_encoder_helper(
        (import_data("./data/bank_data.csv"),
         ['Gender',
          'Education_Level',
          'Marital_Status',
          'Income_Category'])
    )

    test_encoder_helper((import_data("./data/bank_data.csv"),
                        ['Gender',
                         'Education_Level',
                         'Marital_Status',
                         'Income_Category',
                         'Card_Category']))

    test_train_models(perform_feature_engineering(pytest.df, response='Churn'))
