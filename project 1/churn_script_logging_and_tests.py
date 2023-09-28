import logging
import pytest
from churn_library import *


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


@pytest.fixture(scope="module")
def path():
    return "./data/bank_data.csv"


@pytest.fixture(scope="module")
def dataframe(path):
    return import_data(path)


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
    cat_features = request.param
    data = pytest.df.copy()
    return data, cat_features


@pytest.fixture(scope="module")
def input_train():
    data = pytest.df.copy()
    return perform_feature_engineering(data)


@pytest.mark.parametrize("filename",
                         ["./data/bank_data.csv",
                          "./data/no_file.csv"])
def test_import(filename):
    two_test_level = False

    try:
        data = import_data(filename)
        logging.info(f"Testing import_data from file: {filename} - SUCCESS")
        pytest.df = data
        two_test_level = True

    except FileNotFoundError:
        logging.error(
            f"Testing import_data from file: {filename}: The file wasn't found")

    if two_test_level:
        try:
            assert data.shape[0] > 0
            assert data.shape[1] > 0
            logging.info(f"Returned dataframe with shape: {data.shape}")

        except AssertionError:
            logging.error("The file doesn't appear to have rows and columns")


def test_eda():
    data = pytest.df

    try:
        perform_eda(data)
        logging.info("Testing perform_eda - SUCCESS")

    except Exception as err:
        logging.error(f"Testing perform_eda failed - Error type {type(err)}")


def test_encoder_helper(encoder_params):
    two_test_level = False
    data, cat_features = encoder_params

    try:
        newdf = encoder_helper(data, cat_features)
        logging.info(f"Testing encoder_helper with {cat_features} - SUCCESS")
        two_test_level = True

    except KeyError:
        logging.error(
            f"Testing encoder_helper with {cat_features} failed: Check for categorical features not in the dataset")

    except Exception as err:
        logging.error(
            f"Testing encoder_helper failed - Error type {type(err)}")

    if two_test_level:
        try:
            assert newdf.select_dtypes(include='object').columns.tolist() == []
            logging.info("All categorical columns were encoded")

        except AssertionError:
            logging.error(
                "At least one categorical column was NOT encoded - Check categorical features submitted")


def test_perform_feature_engineering():
    two_test_level = False
    try:
        data = pytest.df
        X_train, X_test, y_train, y_test = perform_feature_engineering(data)
        logging.info("Testing perform_feature_engineering - SUCCESS")
        two_test_level = True

    except Exception as err:
        logging.error(
            f"Testing perform_feature_engineering failed - Error type {type(err)}")

    if two_test_level:
        try:
            assert X_train.shape[0] > 0
            assert X_train.shape[1] > 0
            assert X_test.shape[0] > 0
            assert X_test.shape[1] > 0
            assert y_train.shape[0] > 0
            assert y_test.shape[0] > 0
            logging.info(
                f"perform_feature_engineering returned Train / Test set of shape {X_train.shape} {X_test.shape}")

        except AssertionError:
            logging.error(
                "The returned train / test datasets do not appear to have rows and columns")


def test_train_models(input_train):
    try:
        train_models(*input_train)
        logging.info("Testing train_models: SUCCESS")
    except Exception as err:
        logging.error(f"Testing train_models failed - Error type {type(err)}")


if __name__ == "__main__":
    test_import("./data/bank_data.csv")
    test_import("./data/no_file.csv")
    test_perform_feature_engineering(import_data("./data/bank_data.csv"))
    test_encoder_helper(import_data("./data/bank_data.csv"),
                        ['Gender',
                         'Education_Level',
                         'Marital_Status',
                         'Income_Category'])
    test_encoder_helper(import_data("./data/bank_data.csv"),
                        ['Gender',
                         'Education_Level',
                         'Marital_Status',
                         'Income_Category',
                         'Card_Category'])
