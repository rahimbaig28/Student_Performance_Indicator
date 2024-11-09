import sys
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifact', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation.
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))  # for categorical data
                ]
            )

            logging.info(f"categorical columns: {categorical_columns}")
            logging.info(f"numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Print columns to ensure they are as expected
            logging.info("Train DataFrame columns: %s", train_df.columns.tolist())
            logging.info("Test DataFrame columns: %s", test_df.columns.tolist())

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Check for the target column in train and test data
            if target_column_name not in train_df.columns:
                logging.error(f"Target column '{target_column_name}' is missing from train_df")
                raise CustomException(f"Target column '{target_column_name}' is missing from train_df", sys)
            if target_column_name not in test_df.columns:
                logging.error(f"Target column '{target_column_name}' is missing from test_df")
                raise CustomException(f"Target column '{target_column_name}' is missing from test_df", sys)

            # Check for numerical columns in train data
            missing_numerical_columns = [col for col in numerical_columns if col not in train_df.columns]
            if missing_numerical_columns:
                logging.error(f"Missing numerical columns in train_df: {missing_numerical_columns}")
                raise CustomException(f"Missing numerical columns in train_df: {missing_numerical_columns}", sys)

            # Proceed with data transformation
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing data")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path,
            )

        except Exception as e:
            logging.error("Error in data transformation: %s", str(e))
            raise CustomException(e, sys)
