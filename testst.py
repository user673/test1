import os
import re
import ast
import json
import math
import time
import typing
from enum import StrEnum, auto
from collections import defaultdict
from functools import partial
from typing import Callable, List, Tuple, Union

import mlflow
import xgboost
import numpy as np
import seaborn as sns
from addict import Dict
import matplotlib.pyplot as plt
from pandas import concat as pd_concat
import pandas as pd
from pyspark.sql import types as T
from pyspark.sql import functions as F
import pyspark
from pyspark.ml import Pipeline, Estimator, Transformer
from pyspark.ml.evaluation import Evaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from pyspark.sql.column import Column as SparkColumn
from pyspark.sql.functions import PandasUDFType, concat_ws, pandas_udf
from dateutil.relativedelta import relativedelta
from hyperopt import STATUS_OK, fmin, hp, space_eval, tpe
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json
from synapse.ml.lightgbm import LightGBMRegressor
from xgboost.spark import SparkXGBRegressor
from databricks.sdk.runtime import dbutils, spark

from Scripts.helpers.connections import adl_connection_str
from Scripts.helpers.metrics import (
    compute_metrics_by_pair,
    compute_mml_metrics,
    wmape_total,
)


class Encoding(StrEnum):
    """
    Defines different strategies for encoding categorical data.
    """
    LABEL_ENCODING = auto()
    ONE_HOT_ENCODING = auto()


def get_config_from_adl(config_path) -> Dict:
    """
    Load and parse DS config from a file in Azure Data Lake.
    
    Args:
        config_path (str): The path to the config file on the lake
        
    Returns:
        Dict: Parsed config as a dictionary
    """
    config = dbutils.fs.head(
        "{0}{1}".format(adl_connection_str, config_path),
        1_048_576 # The maximum number of bytes to read from the file.
                  # The default is 65,536 bytes (64 KB). For now, it is 1MB.
    )
    config = Dict(json.loads(config))
    return config


def create_logs(
    result: pyspark.sql.dataframe.DataFrame, config: Dict, step_name: str
) -> Tuple[Dict, int, str]:
    """
    This function create logs: counts pairs, saves dataframe with pairs

    Args:
        result (pyspark.sql.dataframe.DataFrame): input pyspark dataframe
        config (Dict): configuration dict
        step_name (str): name of current pipeline module

    Returns:
        Tuple[Dict, int, str]: config, number of pairs, log_path
    """

    # Count number of pairs
    count_pairs = result.count()

    # Create path to save logs
    log_path = "{0}{1}".format(
        config.params.planning_period_id_path, "{0}_logs.csv".format(step_name)
    )

    # Add counts and path to config
    config.logs[config.params.train_score_suffix][step_name]["log_path"] = log_path
    config.logs[config.params.train_score_suffix][step_name]["count"] = count_pairs

    # Save all results
    result.repartition(1).write.mode("overwrite").option(
        "encoding", "windows-1251"
    ).option("header", "true").csv("{0}{1}".format(adl_connection_str, log_path))

    return config, count_pairs, log_path


def str2bool(variable: str) -> bool:
    """
    Fucntion for convertion str bollean value to bool type

    Args:
        variable (str): string representation of bool

    Return:
        boolean variable
    """
    return variable.lower() in ("yes", "true", "1")


def concatenate_values_for_product_business_levels_hierarchy(
    config: Dict,
    splitted_df: pyspark.sql.dataframe.DataFrame,
    product_levels: bool = True,
    business_levels: bool = True,
) -> pyspark.sql.dataframe.DataFrame:
    """
    Function to concatenate all product and business levels to hierarchical columns for pandas dfs

    Args:
        config (str): config dict eith all params
        splitted_df (pyspark.sql.dataframe.DataFrame):  dataset in pandas format with product and business levels in different columns
        product_levels (bool): if product levels should be concatenated
        business_levels (bool): if business levels should be concatenated

    Returns:
      pyspark.sql.dataframe.DataFrame: dataframe with concatenated column with all levels
    """

    concatenated_df = splitted_df

    if product_levels:
        # if there are missing values for levels we should fill them with any value to be able to concatenate them
        concatenated_df = concatenated_df.fillna(
            "nan", subset=config.params.product_levels
        )

        levels_list = config.params.product_levels

        # Concatenate values for each level in levels_list
        concatenated_df = concatenated_df.withColumns(
            {
                level_name: F.concat_ws(
                    "[_]", *levels_list[: levels_list.index(level_name) + 1]
                )
                for level_name in levels_list
            }
        )

        # Create values for "Product_ID"
        concatenated_df = concatenated_df.withColumn(
            "Product_ID", F.col(config.params.product_level)
        )

    if business_levels:
        # if there are missing values for levels we should fill them with any value to be able to concatenate them
        concatenated_df = concatenated_df.fillna(
            "nan", subset=config.params.business_levels
        )

        levels_list = config.params.business_levels

        # Concatenate values for each level in levels_list
        concatenated_df = concatenated_df.withColumns(
            {
                level_name: F.concat_ws(
                    "[_]", *levels_list[: levels_list.index(level_name) + 1]
                )
                for level_name in levels_list
            }
        )

        # Create values for "SalesTypeId"
        concatenated_df = concatenated_df.withColumn(
            "SalesTypeId", F.col(config.params.business_level)
        )

    return concatenated_df


def split_values_for_product_business_levels_hierarchy(
    config: Dict,
    concatenated_df: pyspark.sql.dataframe.DataFrame,
    product_levels: bool = True,
    business_levels: bool = True,
) -> pyspark.sql.dataframe.DataFrame:
    """
    Function to concatenate all product and business levels to hierarchical columns for pandas dfs

    Args:
        config (str): config dict with all params
        concatenated_df (pyspark.sql.dataframe.DataFrame):  dataset in pandas format with product and business levels in different columns
        product_levels (bool): if product levels should be concatenated
        business_levels (bool): if business levels should be concatenated

    Returns:
      pyspark.sql.dataframe.DataFrame: dataframe with concatenated column with all levels
    """
    splitted_df = concatenated_df
    if product_levels:
        splitted_df = splitted_df.withColumn(
            "Product_ID_splitted", F.split("Product_ID", r"\[\_\]")  # Fixed
        )
        for i in range(len(config.params.product_levels)):
            splitted_df = splitted_df.withColumn(
                config.params.product_levels[i], splitted_df["Product_ID_splitted"][i]
            )
    if business_levels:
        splitted_df = splitted_df.withColumn(
            "SalesTypeId_splitted", F.split("SalesTypeId", r"\[\_\]")  # Fixed
        )
        for i in range(len(config.params.business_levels)):
            splitted_df = splitted_df.withColumn(
                config.params.business_levels[i], splitted_df["SalesTypeId_splitted"][i]
            )

    return splitted_df


def get_factors_df(
    data: pyspark.sql.dataframe.DataFrame, model_name: str, timestamp: str
) -> pyspark.sql.dataframe.DataFrame:
    """
    This function group factors by VBB and return dataframe with mapping

    Args:
        data (pyspark.sql.dataframe.DataFrame): data with factors
        model_name (str): name of model
        timestamp (str): modeling timestamp

    Returns:
        pyspark.sql.dataframe.DataFrame: dataframe with mapping factor name  model name and timestamp
    """

    result_df = (
        spark.createDataFrame(data=data.columns, schema="string")
        .withColumnRenamed("value", "factor_name")
        .withColumn("model_name", F.lit(model_name))
        .withColumn("timestamp", F.lit(timestamp))
    )

    return result_df


def get_key_by_value_as_element_of_list(
    value: str, dictionary: dict[str : List[str]]
) -> str:
    """
    This is a helper function that returns the dictionary key of corresponding value,
    which is the list and contains the input 'value' parameter.
    Args:
        value (str): input string value, which is the element of value list, for which we need to get corresponding key in dictionary
        dictionary (dict[str: List[str]]): the input dictionary where we need to find the key.

    Returns:
        str: the needed string key in dictionary
    """

    key = list(dictionary.keys())[
        list(dictionary.values()).index(
            list(filter(lambda values_list: value in values_list, dictionary.values()))[
                0
            ]
        )
    ]
    return key


def get_VBB_for_factor_name(factor_name: str, config: Dict) -> Union[str, None]:
    """
    This function returns the correspodning Volume Bulding Block (VBB) name for input factor name.
    Args:
        factor_name (str): string factor name, that was generated after FE stage
        config (Dict): configuration dictionary with all params

    Returns:
        Union[str, None]: the needed string key in dictionary
    """
    try:
        # Get corresponding DS feature group of feature name
        DS_feature_group = get_key_by_value_as_element_of_list(
            factor_name, config.params.DS_feature_groups_to_features_mapping
        )

        # Get corresponding VBB name of DS feature group
        VBB_name = get_key_by_value_as_element_of_list(
            DS_feature_group, config.params.VBBs_to_DS_feature_groups_mapping
        )

        return VBB_name
    except:
        # We fill unknown VBB_name as NULL if we don't get information about DS_feature_group of input feature name or
        # if we don't know how to map DS_feature_group of input feature name to VBB_name
        return None


def edit_VBB_name_due_to_business_requirements(
    factor_name: str, vbb_name: str, config: Dict
) -> str:
    """
    This function edits the VBB name for factor name (changes its VBB) due to some business requirements.
    Args:
        factor_name (str): string factor name, that was generated after FE stage
        vbb_name (str): corresponding VBB name for factor_name
        config (Dict): configuration dictionary with all params

    Returns:
        str: updated VBB name for factor name
    """
    if "grouped_by" in factor_name or "pivoted" in factor_name:
        return "CANNIBALIZATION"
    elif config.params.price_column in factor_name:
        return "PRICE EFFECT"
    elif config.params.target_name_cleaned in factor_name:
        return "BASELINE"
    elif config.params.promo_discount_percent_column in factor_name:
        return "PROMO"
    elif "_PromoDuration" in factor_name:
        if config.params.regular_promo in factor_name:
            return "BASELINE"
        else:
            return "PROMO"
    else:
        # If factor_name doesn't belong to any additional business requirements then we return its default VBB name
        return vbb_name


def get_factors_names_and_VBBs_dataframe(
    data_after_fs: pyspark.sql.dataframe.DataFrame,
    config: Dict,
    model_name: str,
) -> pyspark.sql.dataframe.DataFrame:
    """
    This function generates dataframe with Volume Building BLocks (VBB) and their corresponding features lists
    after FS stage.
    Args:
        data_after_fs (pyspark.sql.dataframe.DataFrame): dataframe with generated features after FE and filtered features after FS
        config (Dict): configuration dictionary with all params
        model_name (str): ML model name (lgbm|tft|xgboost|baseline)

    Returns:
        pyspark.sql.dataframe.DataFrame: dataframe with Volume Building BLocks (VBB) and their corresponding features lists
    """

    features_with_VBB_name_list = [
        (feature_config["feature_name"], VBB_name)
        for VBB_name in config.features_config.keys()
        for feature_group in config.features_config[VBB_name]
        for feature_config in config.features_config[VBB_name][feature_group]
    ]

    # Generate dataframe with factor names and VBBs from feature constructor
    factors_df_from_features_constructor = spark.createDataFrame(
        data=features_with_VBB_name_list,
        schema=["factor_name", "VBB_name"],
    )

    # Generate dataframe with factor names, model name and modeling timestamp from data after FS
    factors_df_after_fs = get_factors_df(
        data_after_fs, model_name, config.params.timestamp_modeling
    )

    # Join dataframes with factor names from real data after FS and from features constructor to intersect
    # features between 2 dataframes.
    joined_factors_df = factors_df_from_features_constructor.join(
        factors_df_after_fs, how="inner", on="factor_name"
    )

    # Edit some VBB names for features due to business requirements.
    final_df = joined_factors_df.withColumn(
        "VBB_name",
        F.udf(
            lambda factor_name_column, VBB_name_column: edit_VBB_name_due_to_business_requirements(
                factor_name_column, VBB_name_column, config
            ),
        )("factor_name", "VBB_name"),
    )

    return final_df


def get_historical_weight(
    config: Dict, data: pyspark.sql.dataframe.DataFrame, agg_columns: list
) -> pyspark.sql.dataframe.DataFrame:
    """Function for calculating weight column, based on historical value
        old rows will have smaller weight than new ones

    Args:
        config (Dict): config with all paramsp
        data (pyspark.sql.dataframe.DataFrame): input dataset with historical data
        agg_columns (list): list of columns to identificate one sales row

    Returns:
        pyspark.sql.dataframe.DataFrame: dataset plus one column with historical weight
    """

    window_custom_ordered = (
        Window()
        .partitionBy(
            *agg_columns,
        )
        .orderBy(config.params.date_column)
    )
    window_custom = Window().partitionBy(*agg_columns)
    data = data.withColumn(
        "row_number_in_category", F.row_number().over(window_custom_ordered)
    )
    data = data.withColumn(
        "normalize", F.max("row_number_in_category").over(window_custom)
    )
    data = data.withColumn(
        "normalized_value", F.col("row_number_in_category") / F.col("normalize")
    )

    data = data.withColumn(
        "normalize_sum", F.sum("normalized_value").over(window_custom)
    )
    sd = data.withColumn(
        "Evenly_Growing_Column", F.col("normalized_value") / F.col("normalize_sum")
    )
    return sd.drop(
        "normalized_value", "normalize_sum", "normalize", "row_number_in_category"
    )


def get_turnover_weight(
    config: Dict, data: pyspark.sql.dataframe.DataFrame, agg_columns: list
) -> pyspark.sql.dataframe.DataFrame:
    """Function for calculating weight column, based on turnover value

    Args:
        config (Dict): config with all params
        data (pyspark.sql.dataframe.DataFrame): input dataset with historical data
        agg_columns (list): list of columns to identificate one sales row

    Returns:
        pyspark.sql.dataframe.DataFrame: dataset plus one column with turnover weight
    """

    window_custom_ordered = (
        Window()
        .partitionBy(
            *agg_columns,
        )
        .orderBy(config.params.date_column)
    )
    data = data.withColumn(
        "turnover",
        F.col(config.params.target_name_cleaned)
        * F.col(config.params.price_column + "_min"),
    )
    data = data.withColumn(
        "mult_turnover_target",
        F.col(config.params.target_name_cleaned) * F.col("turnover"),
    )

    data = data.withColumn(
        "sum_turnover_mult", F.sum("mult_turnover_target").over(window_custom_ordered)
    )
    data = data.withColumn(
        "sum_turnover", F.sum("turnover").over(window_custom_ordered)
    )

    sd = data.withColumn("average", F.col("sum_turnover_mult") / F.col("sum_turnover"))
    return sd.drop("sum_turnover", "sum_turnover_mult", "mult_turnover_target")


def prepare_turnover_weighted_forecast(
    config: Dict, data_test: pyspark.sql.dataframe.DataFrame, model: str
) -> pyspark.sql.dataframe.DataFrame:
    """
    This function generate simple average forecast
    for data test based on historical sales from data train.

    Args:
        config (Dict): configuration dict
        data_train (pyspark.sql.dataframe.DataFrame): dataset with historical sales
        data_test (pyspark.sql.dataframe.DataFrame): dataset to make forecast for
        model (str):  model name to proceed
    Returns:
        pyspark.sql.dataframe.DataFrame: pyspark dataframe with simple avg forecast
    """
    if model == "tft":
        # Read train data
        save_path_result_train = "{0}_{1}_{2}_{3}_{4}_{5}{6}".format(
            config.params.pipeline_data_name,
            "agg",
            config.params.features_data_path,
            config.params.feature_selection_path,
            "train",
            "tft",
            config.params.data_path_res,
        )

        data_train = spark.read.parquet(
            "{0}{1}{2}".format(
                adl_connection_str,
                config.params.train_planning_period_id_path,
                save_path_result_train,
            )
        )
        print(
            f"[INFO] Path for read data train table command: {save_path_result_train}"
        )

    if model == "lgbm":
        # Read train data
        save_path_result_train = "{0}_{1}_{2}_{3}_{4}{5}".format(
            config.params.pipeline_data_name,
            "agg",
            config.params.features_data_path,
            config.params.feature_selection_path,
            "train",
            config.params.data_path_res,
        )

        data_train = spark.read.parquet(
            "{0}{1}{2}".format(
                adl_connection_str,
                config.params.train_planning_period_id_path,
                save_path_result_train,
            )
        )
        print(
            f"[INFO] Path for read data train table command: {save_path_result_train}"
        )

    if model == "xgboost":
        # Read train data
        config.params.aggregation_data_path = "agg"
        save_path_result_train = "{0}_{1}_{2}_{3}_{4}{5}".format(
            config.params.pipeline_data_name,
            "agg",
            config.params.features_data_path,
            config.params.feature_selection_path,
            "train",
            config.params.data_path_res,
        )

        data_train = spark.read.parquet(
            "{0}{1}{2}".format(
                adl_connection_str,
                config.params.train_planning_period_id_path,
                save_path_result_train,
            )
        )
        print(
            f"[INFO] Path for read data train table command: {save_path_result_train}"
        )

    # we will create concatenated product level for more correct forecasting for data train AND data_test
    # if there are missing values for levels we should fill them with any value to be able to concatenate them
    data_train = data_train.fillna(
        "nan", subset=[config.params.simplified_newbies_product_level]
    )

    # Concatenate values for "config.params.simplified_newbies_product_level"
    test_levels = config.params.product_levels[
        : (config.params.product_levels.index(config.params.product_level))
    ]
    data_train = data_train.withColumn(
        config.params.simplified_newbies_product_level, concat_ws("[_]", *test_levels)
    )
    data_train = get_turnover_weight(
        config, data_train, [config.params.simplified_newbies_product_level]
    )

    simplified_newbies_product_level_agg = data_train.groupBy(
        config.params.simplified_newbies_product_level
    ).agg(F.avg("average").alias("Forecast"))

    # if there are missing values for levels we should fill them with any value to be able to concatenate them
    data_test = data_test.fillna(
        "nan", subset=[config.params.simplified_newbies_product_level]
    )

    # Concatenate values for "config.params.simplified_newbies_product_level"
    data_test = data_test.withColumn(
        config.params.simplified_newbies_product_level, concat_ws("[_]", *test_levels)
    )

    # join simplified forecast for filtered data test
    data_test_forecast = data_test.join(
        simplified_newbies_product_level_agg,
        on=[config.params.simplified_newbies_product_level],
        how="left",
    )

    # preapre final dataset with newbies forecast
    data_test_forecast = data_test_forecast.withColumn("model_name", F.lit(model))
    # Change types and replace negative values of forecast by zeroes
    data_test_forecast = data_test_forecast.withColumn(
        config.params.forecast_column,
        F.when(data_test_forecast[config.params.forecast_column] < 0, 0).otherwise(
            F.col(config.params.forecast_column)
        ),
    )
    data_test_forecast = data_test_forecast.withColumn(
        config.params.newbie_column, F.lit("averages")
    )

    return data_test_forecast


def prepare_simpe_weighted_forecast(
    config: Dict, data_test: pyspark.sql.dataframe.DataFrame, model: str
) -> pyspark.sql.dataframe.DataFrame:
    """
    This function generate simple average forecast
    for data test based on historical sales from data train.

    Args:
        config (Dict): configuration dict
        data_test (pyspark.sql.dataframe.DataFrame): dataset to make forecast for
        model (str): model name

    Returns:
        pyspark.sql.dataframe.DataFrame: pyspark dataframe with simple avg forecast
    """

    if model in ["tft", "baseline"]:

        # Fool proof that no file will be saved with whitespace in naming
        forecasted_product_level = config.params.product_level.replace(" ", "")
        forecasted_business_level = config.params.business_level.replace(" ", "")

        # Change date level name to follow name conventions for day date level
        forecasted_date_level = (
            config.params.date_level if config.params.date_level != "Day" else "Dai"
        )

        # Read train data
        save_path_result_train = f"Sales{forecasted_date_level}ly_{forecasted_product_level}_{forecasted_business_level}.parquet"

    if model in ["lgbm", "xgboost"]:
        # Read train data
        save_path_result_train = "{0}_{1}_{2}_{3}_{4}{5}".format(
            config.params.pipeline_data_name,
            config.params.aggregation_data_path,
            config.params.features_data_path + "_cannibalization",
            config.params.date_level,
            "train",
            config.params.data_path_res,
        )

    data_train = spark.read.parquet(
        "{0}{1}{2}".format(
            adl_connection_str,
            config.params.train_planning_period_id_path,
            save_path_result_train,
        )
    )
    print(f"[INFO] Path for read data train table command: {save_path_result_train}")

    # we will create concatenated product level for more correct forecasting
    # if there are missing values for levels we should fill them with any value to be able to concatenate them
    data_train = data_train.fillna(
        "nan", subset=[config.params.simplified_newbies_product_level]
    )

    # Concatenate values for "config.params.simplified_newbies_product_level"
    test_levels = config.params.product_levels[
        : (config.params.product_levels.index(config.params.product_level))
    ]
    data_train = data_train.withColumn(
        config.params.simplified_newbies_product_level, concat_ws("[_]", *test_levels)
    )

    data_train = get_historical_weight(
        config, data_train, [config.params.simplified_newbies_product_level]
    )
    data_train = data_train.withColumn(
        "new_target",
        F.col(config.params.target_name_cleaned) * F.col("Evenly_Growing_Column"),
    )

    simplified_newbies_product_level_agg = data_train.groupBy(
        config.params.simplified_newbies_product_level
    ).agg(F.sum("new_target").alias("Forecast"))

    # if there are missing values for levels we should fill them with any value to be able to concatenate them
    data_test = data_test.fillna(
        "nan", subset=[config.params.simplified_newbies_product_level]
    )

    # Concatenate values for "config.params.simplified_newbies_product_level"
    data_test = data_test.withColumn(
        config.params.simplified_newbies_product_level, concat_ws("[_]", *test_levels)
    )

    # join simplified forecast for filtered data test
    data_test_forecast = data_test.join(
        simplified_newbies_product_level_agg,
        on=[config.params.simplified_newbies_product_level],
        how="left",
    )

    # preapre final dataset with newbies forecast
    data_test_forecast = data_test_forecast.withColumn("model_name", F.lit(model))
    # Change types and replace negative values of forecast by zeroes
    data_test_forecast = data_test_forecast.withColumn(
        config.params.forecast_column,
        F.when(data_test_forecast[config.params.forecast_column] < 0, 0).otherwise(
            F.col(config.params.forecast_column)
        ),
    )
    data_test_forecast = data_test_forecast.withColumn(
        config.params.newbie_column, F.lit("averages")
    )

    return data_test_forecast


def prepare_simpe_forecast(
    config: Dict, data_test: pyspark.sql.dataframe.DataFrame, model: str
) -> pyspark.sql.dataframe.DataFrame:
    """
    This function generate simple average forecast
    for data test based on historical sales from data train.

    Args:
        config (Dict): configuration dict
        data_test (pyspark.sql.dataframe.DataFrame): dataset to make forecast for
        model (str): model name

    Returns:
        pyspark.sql.dataframe.DataFrame: pyspark dataframe with simple avg forecast
    """
    if model == "tft":  # in config.params.model_family:
        # Read train data
        save_path_result_train = "{0}_{1}_{2}_{3}_{4}_{5}{6}".format(
            config.params.pipeline_data_name,
            "agg",
            config.params.features_data_path,
            config.params.feature_selection_path,
            "train",
            "tft",
            config.params.data_path_res,
        )

        data_train = spark.read.parquet(
            "{0}{1}{2}".format(
                adl_connection_str,
                config.params.train_planning_period_id_path_unique["tft"],
                save_path_result_train,
            )
        )
        print(
            f"[INFO] Path for read data train table command: {save_path_result_train}"
        )

    if model == "lgbm":
        # Read train data
        save_path_result_train = "{0}_{1}_{2}_{3}_{4}{5}".format(
            config.params.pipeline_data_name,
            "agg",
            config.params.features_data_path,
            config.params.feature_selection_path,
            "train",
            config.params.data_path_res,
        )

        data_train = spark.read.parquet(
            "{0}{1}{2}".format(
                adl_connection_str,
                config.params.train_planning_period_id_path_unique["lgbm"],
                save_path_result_train,
            )
        )
        print(
            f"[INFO] Path for read data train table command: {save_path_result_train}"
        )

    if model == "xgboost":
        # Read train data
        save_path_result_train = "{0}_{1}_{2}_{3}_{4}{5}".format(
            config.params.pipeline_data_name,
            "agg",
            config.params.features_data_path,
            config.params.feature_selection_path,
            "train",
            config.params.data_path_res,
        )

        data_train = spark.read.parquet(
            "{0}{1}{2}".format(
                adl_connection_str,
                config.params.train_planning_period_id_path_unique["xgboost"],
                save_path_result_train,
            )
        )
        print(
            f"[INFO] Path for read data train table command: {save_path_result_train}"
        )

    # we will create concatenated product level for more correct forecasting
    # if there are missing values for levels we should fill them with any value to be able to concatenate them
    data_train = data_train.fillna(
        "nan", subset=[config.params.simplified_newbies_product_level]
    )

    # Concatenate values for "config.params.simplified_newbies_product_level"
    test_levels = config.params.product_levels[
        : (config.params.product_levels.index(config.params.product_level))
    ]
    data_train = data_train.withColumn(
        config.params.simplified_newbies_product_level, concat_ws("[_]", *test_levels)
    )

    simplified_newbies_product_level_agg = data_train.groupBy(
        config.params.simplified_newbies_product_level
    ).agg(F.avg(config.params.target_name_cleaned).alias("Forecast"))

    # if there are missing values for levels we should fill them with any value to be able to concatenate them
    data_test = data_test.fillna(
        "nan", subset=[config.params.simplified_newbies_product_level]
    )

    # Concatenate values for "config.params.simplified_newbies_product_level"
    data_test = data_test.withColumn(
        config.params.simplified_newbies_product_level, concat_ws("[_]", *test_levels)
    )

    # join simplified forecast for filtered data test
    data_test_forecast = data_test.join(
        simplified_newbies_product_level_agg,
        on=[config.params.simplified_newbies_product_level],
        how="left",
    )

    # preapre final dataset with newbies forecast
    data_test_forecast = data_test_forecast.withColumn("model_name", F.lit(model))
    # Change types and replace negative values of forecast by zeroes
    data_test_forecast = data_test_forecast.withColumn(
        config.params.forecast_column,
        F.when(data_test_forecast[config.params.forecast_column] < 0, 0).otherwise(
            F.col(config.params.forecast_column)
        ),
    )
    data_test_forecast = data_test_forecast.withColumn(
        config.params.newbie_column, F.lit("averages")
    )

    return data_test_forecast


def display_importances(
    feature_importance_df: pd.DataFrame,
    model_name: str,
    n_features: int,
    window: str = "final",
) -> None:
    """Function for making bar plot with top 50 model's feature importance and saving in to file

    Args:
        feature_importance_df (pd.DataFrame): dataframe with importances for every feature
        model_name (str): name of ml model used for training
        val (int): number of validation window
        n_features (int): number of features to display

    """

    data = feature_importance_df.sort_values(by="importance", ascending=False).head(
        n_features
    )

    plt.figure(figsize=(15, 15))
    sns.barplot(y=data.feature, x=data.importance, orient="h")
    plt.title("{} Features ".format(model_name))
    plt.tight_layout()
    plt.savefig("importance_val_{0}.png".format(window))
    mlflow.log_artifact("importance_val_{0}.png".format(window))
    os.remove("importance_val_{0}.png".format(window))


def get_categorical_columns(
    data: pyspark.sql.dataframe.DataFrame, config: Dict
) -> List[str]:
    """
    This function check data types of all columns and return
    list of categorical ones

    Args:
        data (pyspark.sql.dataframe.DataFrame): input dataset with columns to analyse
        config (Dict): configuration dict

    Returns:
        List(str): list of categorical columns
    """
    categorical_columns_list = [
        column_name
        for column_name, column_type in data.dtypes
        if column_type.startswith("string")
    ]
    return categorical_columns_list


def get_not_categorical_columns(
    data: pyspark.sql.dataframe.DataFrame, config: Dict
) -> List[str]:
    """
    This function check data types of all columns and return
    list of not categorical ones

    Args:
        data (pyspark.sql.dataframe.DataFrame): input dataset with columns to analyse
        config (Dict): configuration dict

    Returns:
        List(str): list of not categorical columns
    """
    categorical_columns = get_categorical_columns(data, config)
    columns_to_not_include = categorical_columns
    numerical_columns_list = list(set(data.columns) - set(columns_to_not_include))
    return numerical_columns_list


def prepare_data(
    config: Dict,
    mode: str,
    train_data: pyspark.sql.dataframe.DataFrame = None,
    run_id=None,
    test_data: pyspark.sql.dataframe.DataFrame = None,
) -> pyspark.sql.dataframe.DataFrame:
    """
    This function prepares dataset to needed format for baseline model.
    Perform encodind and feature creation

    Args:

        config (Dict): configuration dict
        mode (str): type of run (val, score or train)
        train_data (pyspark.sql.dataframe.DataFrame): train dataset for modeling
        test_data (pyspark.sql.dataframe.DataFrame): test dataset for scoring

    Returns:
        pyspark.sql.dataframe.DataFrame: dataset in needed format
    """
    if mode != "score":
        categorical_columns = get_categorical_columns(train_data, config)
        categoricalSlotNames = [
            categorical_column + "_indexed"
            for categorical_column in categorical_columns
        ]

        all_other_numerical_features = get_not_categorical_columns(train_data, config)

        category_indexer = StringIndexer(
            inputCols=categorical_columns,
            outputCols=categoricalSlotNames,
            handleInvalid="keep",
        )
        print(
            "[INFO] All columns for train",
            categoricalSlotNames + all_other_numerical_features,
        )
        feature_assembler = VectorAssembler(
            inputCols=categoricalSlotNames + all_other_numerical_features,
            outputCol="features",
            handleInvalid="keep",
        )

        transforming_pipeline = Pipeline(stages=[category_indexer, feature_assembler])

        trained_pipeline = transforming_pipeline.fit(train_data)

        # The default path where the MLflow autologging function stores the model
        if mode == "final":
            mlflow.spark.log_model(
                spark_model=trained_pipeline, artifact_path="model_transform"
            )

        train_data_transformed = trained_pipeline.transform(train_data)
        if mode == "val":
            test_data_transformed = trained_pipeline.transform(test_data)
            return train_data_transformed, test_data_transformed
        return train_data_transformed
    else:
        trained_pipeline = mlflow.spark.load_model(
            "runs:/{}/model_transform".format(run_id)
        )
        test_data_transformed = trained_pipeline.transform(test_data)
        return test_data_transformed


def prepare_forecast(
    df_with_forecast: pyspark.sql.dataframe.DataFrame,
    config: Dict,
    model_name: str = None,
    shap: bool = False,
) -> pyspark.sql.dataframe.DataFrame:
    """
    This function prepares dataset with predictions to needed format

    Args:
        df_with_forecast (pyspark.sql.dataframe.DataFrame): dataset after performmig scoring (with predictions)
        config (Dict): configuration dict
        model_name (str): name of the model
        shap (bool): if it is waterfall run or not

    Returns:
        pyspark.sql.dataframe.DataFrame: dataset in needed format
    """
    # Read needed tables
    Item = spark.read.parquet(
        "{0}{1}".format(
            adl_connection_str, config.params.product_not_hierarchy_catalog_path
        )
    )
    ItemUoM = spark.read.parquet(
        "{0}{1}".format(
            adl_connection_str,
            "{}_USD/{}".format(
                config.params.organization_name, config.params.item_uom_table
            ),
        )
    )
    UoMs = spark.read.parquet(
        "{0}{1}".format(
            adl_connection_str,
            "{}_USD/{}".format(config.params.organization_name, "UoMs.parquet"),
        )
    )

    if model_name is not None:
        df_with_forecast = df_with_forecast.withColumn("model_name", F.lit(model_name))

    # Change types and replace negative values of forecast by zeroes
    df_with_forecast = df_with_forecast.withColumn(
        config.params.forecast_column,
        F.when(F.col(config.params.forecast_column) < 0, 0).otherwise(
            F.col(config.params.forecast_column)
        ),
    )

    # if there are nan in some choosen columns - this should be a newbie (mark this row as newbies=1)
    choosen_cols = [
        col
        for col in df_with_forecast.columns
        if any(elem in col for elem in ["lag", "moving", "expanding"])
    ]
    null_columns_list = [F.col(column).isNull() for column in choosen_cols] + [
        F.lit(False)
    ] * 2
    df_with_forecast = df_with_forecast.withColumn(
        config.params.newbie_column, F.greatest(*null_columns_list).cast("int")
    )

    df_with_UoMs = Item.groupby(config.params.product_levels).agg(
        F.mode(config.params.planning_uom_column).alias(
            config.params.planning_uom_column
        )
    )

    df_with_UoMs = concatenate_values_for_product_business_levels_hierarchy(
        config, df_with_UoMs, business_levels=False
    )

    df_with_forecast = df_with_forecast.join(
        df_with_UoMs.select(
            config.params.product_level, config.params.planning_uom_column
        ),
        on=config.params.product_level,
        how="left",
    )

    df_with_forecast = df_with_forecast.withColumn(
        config.params.planning_aggregate_column, F.lit(config.params.date_level)
    )
    df_with_forecast = df_with_forecast.withColumn(
        config.params.planning_period_start_day_column,
        F.lit(config.params.cycle_opened),
    )
    df_with_forecast = df_with_forecast.withColumn(
        config.params.planning_horizon_column, F.lit(config.params.forecast_period)
    )
    df_with_forecast = df_with_forecast.withColumnRenamed(
        config.params.date_column, config.params.promo_start_date
    )

    if config.params.date_level == "Day":
        df_with_forecast = df_with_forecast.withColumn(
            config.params.promo_end_date, F.col(config.params.promo_start_date)
        )

    elif config.params.date_level == "Week":
        df_with_forecast = df_with_forecast.withColumn(
            config.params.promo_end_date,
            F.date_add(config.params.promo_start_date, days=6),
        )

    elif config.params.date_level == "Month":
        df_with_forecast = df_with_forecast.withColumn(
            config.params.promo_end_date,
            F.date_sub(F.add_months(config.params.promo_start_date, months=1), days=1),
        )

    rounded_df_with_forecast = round_forecast(
        config, df_with_forecast, Item, ItemUoM, UoMs
    )

    # rename column for business and product level
    rounded_df_with_forecast = rounded_df_with_forecast.withColumnRenamed(
        config.params.business_level, "SalesTypeId"
    )
    rounded_df_with_forecast = rounded_df_with_forecast.withColumnRenamed(
        config.params.product_level, "Product_ID"
    )

    df_with_forecast = select_needed_columns_for_resulting_dataset(
        config, rounded_df_with_forecast, shap
    )

    return df_with_forecast


def round_forecast(
    config: Dict,
    forecast: pyspark.sql.dataframe.DataFrame,
    item: pyspark.sql.dataframe.DataFrame,
    item_uom: pyspark.sql.dataframe.DataFrame,
    uoms: pyspark.sql.dataframe.DataFrame,
) -> pyspark.sql.dataframe.DataFrame:
    """
    This function prepares dataset wirh predictions to needed format

    Args:
        config (Dict): configuration dict
        forecast (pyspark.sql.dataframe.DataFrame): dataset after performmig scoring (with predictions)
        item (pyspark.sql.dataframe.DataFrame): UDS dataset with descriprion for each product
        item_uom (pyspark.sql.dataframe.DataFrame): UDS dataset with lifecycles for each product-pair
        uoms (pyspark.sql.dataframe.DataFrame): UDS dataset with uom descriptions

    Returns:
        pyspark.sql.dataframe.DataFrame: dataset in needed format
    """

    # first round to 6 decimal places to skip VERY low predictions as 0
    forecast = forecast.withColumn(
        config.params.forecast_column, F.round(F.col(config.params.forecast_column), 6)
    )
    if config.params.product_level != "Product_ID":
        forecast = forecast.join(
            uoms,
            forecast[config.params.planning_uom_column] == uoms["UoM"],
            how="left",
        ).drop("UoM")

        forecast = forecast.withColumn(
            config.params.forecast_column,
            F.when(
                F.col("Int") == 1, F.round(F.col(config.params.forecast_column))
            ).otherwise(F.col(config.params.forecast_column)),
        )

        forecast = forecast.drop("Int")

    else:

        # get full information about item
        item_full = item_uom.join(
            item.select(
                *config.params.product_levels,
                config.params.planning_uom_column,
                config.params.base_uom_column,
            ),
            on=[config.params.product_level],
            how="left",
        )

        # join UoMs info for Plannign UoM
        item_full = item_full.join(
            uoms.withColumnRenamed("Int", "Int_planning"),
            item_full[config.params.planning_uom_column] == uoms["UoM"],
            how="left",
        ).drop("UoM")

        # join UoMs info for Base UoM
        item_full = item_full.join(
            uoms.withColumnRenamed("Int", "Int_base"),
            item_full[config.params.base_uom_column] == uoms["UoM"],
            how="left",
        ).drop("UoM")

        # select only those lines that represent convertation between plan and base UoMs
        item_full = item_full.filter(
            item_full[config.params.planning_uom_column]
            == item_full[config.params.item_uom_UoM_ID_column]
        )

        # merge forecast with this information
        item_full = concatenate_values_for_product_business_levels_hierarchy(
            config, item_full, business_levels=False
        )

        forecast = forecast.join(
            item_full.select(
                config.params.product_level,
                config.params.base_uom_column,
                config.params.item_uom_quantity_per_UoM_column,
                "Int_planning",
                "Int_base",
            ),
            on=[config.params.product_level],
            how="left",
        )

        forecast = forecast.withColumn(
            config.params.forecast_column,
            F.when(
                (F.col("Int_planning") == 0) & (F.col("Int_base") == 0),
                F.col(config.params.forecast_column),
            )
            .when(
                (F.col("Int_planning") == 1) & (F.col("Int_base") == 1),
                F.round(F.col(config.params.forecast_column), 0),
            )
            .when(
                (F.col("Int_planning") == 0)
                & (F.col("Int_base") == 1)
                & (F.col(config.params.forecast_column) <= 0),
                0,
            )
            .when(
                (F.col("Int_planning") == 0)
                & (F.col("Int_base") == 1)
                & (F.col(config.params.forecast_column) > 0),
                F.col(config.params.item_uom_quantity_per_UoM_column)
                * F.round(
                    F.col(config.params.forecast_column)
                    / F.col(config.params.item_uom_quantity_per_UoM_column)
                ),
            )
            .when(
                (F.col("Int_planning") == 1) & (F.col("Int_base") == 0),
                F.round(F.col(config.params.forecast_column), 0),
            ),
        )

        forecast = forecast.drop(
            config.params.item_uom_quantity_per_UoM_column, "Int_planning", "Int_base"
        )

    return forecast


def select_needed_columns_for_resulting_dataset(
    config: Dict,
    dataset: pyspark.sql.dataframe.DataFrame,
    shap: bool = False,
) -> pyspark.sql.dataframe.DataFrame:
    """Function for selecting needed columns for resulting dataset
    (validation or scoring dataset)

    Args:
        config (Dict): config with all params
        dataset (pyspark.sql.dataframe.DataFrame): input dataset for columns selection
        shap (bool): if it is waterfall run or not

    Returns:
        pyspark.sql.dataframe.DataFrame: full test dataset with only necessary columns
    """

    other_columns_to_select = (
        config.params.other_scoring_final_dataset_columns
        if config.params.scoring_mode
        else config.params.other_validation_final_dataset_columns
    )

    if config.params.scoring_mode:

        # without this if by the end of start_newbies_identification.py for scoring all models we have multiple "complex_scenario" columns simultaneously added
        if "complex_scenario" not in other_columns_to_select:
            other_columns_to_select.append("complex_scenario")

        if shap:
            other_columns_to_select.append("shap_values")

    other_columns_to_select = list(set(other_columns_to_select))
    result_dataset = dataset.select(
        "Product_ID",
        "SalesTypeId",
        *other_columns_to_select,
    )

    return result_dataset


def get_MaxFutureDate() -> str:
    """
    Function to retrieve the maximum future date for dictionaries

    Returns:
        str: The maximum future date.
    """
    return (
        spark.sql(
            """
        SELECT MaxFutureDate 
        FROM silver.MaxFutureDate
        ORDER BY CreatedOn DESC
        LIMIT 1
        """
        )
        .first()[0]
        .strftime("%Y-%m-%d")
    )



class MAPEEvaluator(Evaluator):
    def __init__(self, predictionCol="prediction", labelCol="label"):
        self.predictionCol = predictionCol
        self.labelCol = labelCol

    def _evaluate(self, dataset: pyspark.sql.dataframe.DataFrame) -> float:
        """
        Evaluates the output with proper error handling for edge cases.

        Args:
            dataset (pyspark.sql.dataframe.DataFrame): a dataset that contains labels/observations and predictions

        Returns:
            float: metric
        """
        # First check if dataset is empty
        if dataset.rdd.isEmpty():
            print("ERROR: Empty dataset for MAPE evaluation, can't proceed with tuning")
            raise ValueError(
                f"Empty dataset. Cannot proceed with hyperparameter tuning. "
            )

        # Calculate MAPE only for non-zero labels to avoid division by zero
        dataset_with_calculated_metric = dataset.withColumn(
            "MAPE",
            F.when(
                F.col(self.labelCol) != 0,
                F.abs(
                    (F.col(self.labelCol) - F.col(self.predictionCol)) 
                    / F.col(self.labelCol)
                )
            ).otherwise(
                # For zero labels, use absolute error normalized by a small constant
                F.when(
                    F.col(self.predictionCol) == 0, 
                    0.0  # If both are zero, error is 0
                ).otherwise(
                    1.0  # If label is 0 but prediction is not, assign max error
                )
            )
        )
        
        # Calculate average MAPE
        mape_plain_value = dataset_with_calculated_metric.select(
            F.avg("MAPE").alias("MAPE")
        ).first()["MAPE"]
        
        return mape_plain_value

    def isLargerBetter(self) -> bool:
        """
        Indicates whether the metric returned by :py:meth:`evaluate` should be maximized
        (True, default) or minimized (False).
        A given evaluator may support multiple metrics which may be maximized or minimized.
        """
        return False  # For MAPE, smaller is better


def check_if_file_exists_in_lake_directory_path(file_name, directory_path):
    """
    This function checks is file exist in the specified шdrectory

    Args:
        file_name (str): file name
        directory_path (str): data lake directory path (abfss:/[file_system]@[account_name].dfs.core.windows.net/[path])
    Returns:
        bool: true if file exist, false if not
    """
    files_info = dbutils.fs.ls(directory_path)
    result_of_search = any(
        filter(lambda file_info: file_name == file_info.name, files_info)
    )
    return result_of_search


def check_if_hyperparams_tuning_is_needed(config: Dict) -> Tuple[dict, dict]:
    """
    Function to check if there is a need to perform hyperparameter tuning

    Args:
        config (Dict): config dict eith all params

    Returns:
      Tuple[dict, dict]: dictionary with bool values for each model (to run or not hypertune), if no need in hypertune - second dict has paths for each model ready hyperparams, in other case it's empty
    """
    hyperparams_tune_needed = dict()
    hyperparams_file_paths_dict = dict()

    for model in config.params.model_family:
        if model not in ["tft", "baseline"]:

            target_hyperparams_file_path = "{0}{1}{2}".format(
                adl_connection_str,
                config.params.train_planning_period_id_path,
                config.params.hyperparams_path[model],
            )
            if ((model == "tft") and (config.params.windows_tft == 0)) or (
                (model != "tft") and (config.params.val_windows == 0)
            ):
                # In this case we can not launch hyperparams tuning because of validation windows lack
                hyperparams_tune_needed[model] = False

                # Save empty dict in hyperparams file to use default hyperparams for model training
                empty_hyperparams_dict_json = json.dumps(ast.literal_eval("{}"))
                dbutils.fs.put(
                    target_hyperparams_file_path,
                    str(empty_hyperparams_dict_json),
                    overwrite=True,
                )

            else:
                if config.params.do_hypertune:
                    # We need to launch the hyperparams tuning for model
                    hyperparams_tune_needed[model] = True
                else:
                    uds_root_path = "{0}{1}".format(
                        adl_connection_str, config.params.organization_name + "_USD/"
                    )
                    uds_root_hyperparams_file_path = "{0}{1}".format(
                        uds_root_path, config.params.hyperparams_path[model]
                    )
                    if check_if_file_exists_in_lake_directory_path(
                        file_name=config.params.hyperparams_path[model],
                        directory_path="{0}{1}".format(
                            adl_connection_str,
                            config.params.train_planning_period_id_path,
                        ),
                    ):
                        # Nothing to do - the generated hyperparams file already exists in correct location
                        hyperparams_tune_needed[model] = False
                        hyperparams_file_paths_dict[
                            model
                        ] = target_hyperparams_file_path

                    elif check_if_file_exists_in_lake_directory_path(
                        file_name=config.params.hyperparams_path[model],
                        directory_path=uds_root_path,
                    ):
                        # We need to copy existing hyperparams file from UDS root to correct planning period folder
                        hyperparams_tune_needed[model] = False

                        successful_copy_operation = dbutils.fs.cp(
                            uds_root_hyperparams_file_path,
                            target_hyperparams_file_path,
                            recurse=False,
                        )

                        if successful_copy_operation:
                            hyperparams_file_paths_dict[
                                model
                            ] = target_hyperparams_file_path
                        else:
                            # We need to launch the hyperparams tuning for model explicitly
                            hyperparams_tune_needed[model] = True

                    else:
                        # We need to launch the hyperparams tuning for model
                        hyperparams_tune_needed[model] = True

    return hyperparams_tune_needed, hyperparams_file_paths_dict


def reduce_dataset_size_used_for_hyperparams_tune(
    data_train: pyspark.sql.dataframe.DataFrame, config: Dict
) -> pyspark.sql.dataframe.DataFrame:
    """
    This function filter dataframe for hyperparams tuning. Only part of dataset will be used (for perfomance issues)

    Args:
        data_train (pyspark.sql.dataframe.DataFrame): input dataset with features
        config (Dict): configuration dict

    Returns:
        pyspark.sql.dataframe.DataFrame: dataset with reduced data
    """
    # In this line we choose what will be our target for reducing size of dataset - product levels or business
    # if config.params.use_prods_for_hypertune  == True, we will calculate all distinct values from product level and get config.params.perc_data_hypertune of them for tuning
    # in other case - same thing with business level
    levels_to_use = (
        config.params.product_levels
        if config.params.use_prods_for_hypertune
        else config.params.business_levels
    )

    unique_used_levels = data_train.select(levels_to_use).distinct()

    needed_levels_to_use = unique_used_levels.sample(
        withReplacement=False,
        fraction=(config.params.perc_data_hypertune / 100),
        seed=0,
    )

    data_sampled = needed_levels_to_use.join(data_train, how="inner", on=levels_to_use)

    return data_sampled


def split_data_for_train_test_datasets(
    data: pyspark.sql.dataframe.DataFrame,
    config: Dict,
    val_window: int,
) -> Tuple[pyspark.sql.dataframe.DataFrame, pyspark.sql.dataframe.DataFrame]:
    """
    This function splits data in train/test pair, based on input params such as forecast period,
    max date in input data, cycl open for this planing period and aggregation levels

    Args:
        data (pyspark.sql.dataframe.DataFrame): input dataset with features
        config (Dict): configuration dict
        val_window (int): number of valudation window

    Returns:
        Tuple[pyspark.sql.dataframe.DataFrame, pyspark.sql.dataframe.DataFrame]: train and test dataset
    """
    # limb period is frozen + forecast period, where frozen is period from max date in data to cycle opened (can be 0)
    limb_period_size = config.params.limb_period
    forecast_period_size = config.params.forecast_period

    limb_period = (
        relativedelta(days=limb_period_size)
        if config.params.date_level == "Day"
        else (
            relativedelta(weeks=limb_period_size)
            if config.params.date_level == "Week"
            else (
                relativedelta(months=limb_period_size)
                if config.params.date_level == "Month"
                else (
                    relativedelta(years=limb_period_size)
                    if config.params.date_level == "Year"
                    else None
                )
            )
        )
    )
    forecast_period = (
        relativedelta(days=forecast_period_size)
        if config.params.date_level == "Day"
        else (
            relativedelta(weeks=forecast_period_size)
            if config.params.date_level == "Week"
            else (
                relativedelta(months=forecast_period_size)
                if config.params.date_level == "Month"
                else (
                    relativedelta(years=forecast_period_size)
                    if config.params.date_level == "Year"
                    else None
                )
            )
        )
    )

    max_date_in_input_data, min_date_in_input_data = data.select(
        F.max(config.params.date_column), F.min(config.params.date_column)
    ).first()

    # Last date (ending date) for validation dataset for window N is calculated to simulate real test dataset, so we subtract limb period N-1 times
    # (-1 becouse for 1 window, last validation date is last date in dataset)
    last_date_for_test_part = max_date_in_input_data - limb_period * (val_window - 1)

    # for first date (starting date) in validation we need to substruct forecast period and add some rounding (because of date calculateions in python)
    first_date_for_test_part = (
        last_date_for_test_part - forecast_period + relativedelta(days=1)
    )

    # Lsst date (ending) for training period can be calculated subtructing limb period (frozen + forecast) from last (ending date) of validation perid
    last_date_for_train_part = last_date_for_test_part - limb_period

    # Starting date for train data is just a starting date from all dataset
    first_date_for_train_part = min_date_in_input_data

    train_data = data.filter(
        F.col(config.params.date_column).between(
            first_date_for_train_part, last_date_for_train_part
        )
    )
    test_data = data.filter(
        F.col(config.params.date_column).between(
            first_date_for_test_part, last_date_for_test_part
        )
    )

    return train_data, test_data


def get_databricks_cluster_cpu_data() -> Tuple[int, int, int]:
    """
    This function gets information about running cluster, such as number of workers,
     number of cpu cores per worker and number of cpu cores in all cluster

    Args:

    Returns:
        Tuple[int, int, int]: information about running cluster
    """
    spark_configuration = spark.builder.getOrCreate().sparkContext.getConf()
    WorkersNum = int(
        spark_configuration.get(
            "spark.databricks.clusterUsageTags.clusterTargetWorkers"
        )
    )

    # defaultParallelism attribute of spark context is total number of CPU cores for all workers in cluster
    TotalAvailableCPUCoresInCluster = spark.sparkContext.defaultParallelism
    CPUcoresPerWorker = TotalAvailableCPUCoresInCluster // WorkersNum

    return WorkersNum, CPUcoresPerWorker, TotalAvailableCPUCoresInCluster


def get_columns_info_for_model_training(config, df):
    """
    This function gets list of categorical and non-categorical columns from
    input dataset and renames them to 'indexed' name, for pyspark encoding

    Args:
        config (Dict): configuration dict
        df (pyspark.sql.dataframe.DataFrame): input dataset with features

    Returns:
        Tuple[List[str], List[str], List[str]]: lists of column names
    """
    old_categorical_column_names = get_categorical_columns(df, config)
    numerical_features = get_not_categorical_columns(df, config)
    new_categorical_column_names = [
        categorical_column_name + "_indexed"
        for categorical_column_name in old_categorical_column_names
    ]
    enc_categorical_column_names = [
        categorical_column_name + "_encoded"
        for categorical_column_name in new_categorical_column_names
    ]

    return (
        old_categorical_column_names,
        numerical_features,
        new_categorical_column_names,
        enc_categorical_column_names,
    )


def get_ml_model(
    config: Dict,
    model_name: str,
    hyperparams: dict,
    df: pyspark.sql.dataframe.DataFrame,
    shap: bool,
) -> Union[
    None,
    LightGBMRegressor,
    xgboost.spark.estimator.SparkXGBRegressor,
    LinearRegression,
]:
    """
    This function create model object based on  model name and hyperparametes

    Args:
        config (Dict): configuration dict
        model_name (str): name of the model
        hyperparams (dict): dictionary with hyperparams for each model
        df (pyspark.sql.dataframe.DataFrame): input dataset with features
        shap (bool): if we need to add shap values to model results or not

    Returns:
       Union[None, LightGBMRegressor.LightGBMRegressor, xgboost.spark.estimator.SparkXGBRegressor]: None if model_name is incorrect and model object in other case (lgbm or xgboost)
    """
    if model_name in ['lgbm','xgboost']:
        (
            WORKERS_NUM,
            AVAILABLE_CPU_CORES_PER_WORKER,
            TOTAL_AVAILABLE_CPU_CORES_IN_CLUSTER,
        ) = get_databricks_cluster_cpu_data()
        (
            old_categorical_column_names,
            numerical_features,
            new_categorical_column_names,
            enc_categorical_column_names,
        ) = get_columns_info_for_model_training(config, df)
        
    if hyperparams!={}:
        spark_model_hyperparams = get_hyperparams_for_spark_model(model_name, hyperparams)
    else:
        spark_model_hyperparams = hyperparams # use default hyperparams

    if model_name == "lgbm":
        # Get information aboue categorical features for fixing connection refused error
        # Calculate the number of unique values for each column
        if new_categorical_column_names != []:
            unique_counts = [
                (col, df.select(col).distinct().count())
                for col in old_categorical_column_names
            ]
            # Find the column with the maximum number of unique values. It should not be smaller than 32
            max_unique_col = max(unique_counts, key=lambda x: x[1])
            if max_unique_col[1] // 2 < 32:
                value_maxCatThreshold = 32
            else:
                value_maxCatThreshold = max_unique_col[1]
        else:
            value_maxCatThreshold = 32
        if shap:
            model = LightGBMRegressor(
                numTasks=0,
                # Handle categorical features
                categoricalSlotNames=new_categorical_column_names,
                featuresCol="features",
                labelCol=config.params.target_name_cleaned,
                predictionCol=config.params.forecast_column,
                # Set passThroughArgs to pass exactly lightgbm regressor hyperparams format, not
                # synapseml spark lightgbm regressor hyperparams format
                passThroughArgs="deterministic=true force_row_wise=true pre_partition=true",
                matrixType="dense",
                seed=0,
                verbosity=3,
                maxCatThreshold=value_maxCatThreshold,
                featuresShapCol="shap_values",
                **spark_model_hyperparams,
            )
        else:
            model = LightGBMRegressor(
                numTasks=0,
                # Handle categorical features
                categoricalSlotNames=new_categorical_column_names,
                featuresCol="features",
                labelCol=config.params.target_name_cleaned,
                predictionCol=config.params.forecast_column,
                # Set passThroughArgs to pass exactly lightgbm regressor hyperparams format, not
                # synapseml spark lightgbm regressor hyperparams format
                passThroughArgs="deterministic=true force_row_wise=true pre_partition=true",
                matrixType="dense",
                seed=0,
                verbosity=3,
                maxCatThreshold=value_maxCatThreshold,
                **spark_model_hyperparams,
            )
        

    elif model_name == "xgboost":
        if shap:
            model = SparkXGBRegressor(
                num_workers=TOTAL_AVAILABLE_CPU_CORES_IN_CLUSTER,
                features_col="features",
                label_col=config.params.target_name_final_cleaned,
                prediction_col=config.params.forecast_column,
                validate_parameters=True,
                **spark_model_hyperparams,
                tree_method="hist",
                pred_contrib_col="shap_values",
                seed=0,
            )
        else:
            model = SparkXGBRegressor(
                num_workers=TOTAL_AVAILABLE_CPU_CORES_IN_CLUSTER,
                features_col="features",
                label_col=config.params.target_name_final_cleaned,
                prediction_col=config.params.forecast_column,
                validate_parameters=True,
                **spark_model_hyperparams,
                tree_method="hist",
                seed=0,
            )

    elif model_name == "baseline":
        model = LinearRegression(
            featuresCol="features",
            labelCol=config.params.target_name_cleaned,
            predictionCol=config.params.forecast_column,
        )

    elif model_name == "tft":
        model = None
    else:
        raise Exception(f"Unknown model type to train, got {model_name} model!")

    return model


def train_ml_model(
    model: Estimator,
    train_data: pyspark.sql.dataframe.DataFrame,
    max_retries: int = 3,
    delay: int = 600,
) -> Transformer:
    """
    This function trains the pyspark ml model and return trained model version. If the training fails, it restarts
    `max_retries` times at intervals of `delay` seconds. Delay is necessary to give a time to cluster to set up a missed node.

    Args:
        model (Estimator) - pyspark ml model to train
        train_data (pyspark.sql.dataframe.DataFrame) - train data for model, transformed (stringIndexer + vectorAssembler)
        max_retries (int) - number of training retries. Default is 3 times.
        delay (int) - delay between the retries in seconds. Default is 600 seconds (10 minutes)

    Returns:
       Transformer: trained pyspark ml model
    
    Raises:
        RuntimeError: If the training fails after `max_retries` attempts
    """

    if isinstance(model, LightGBMRegressor):
        # Temporary workaround to handle connection refused error for lgbm model. It will be removed during
        # implementation the approach of model training using ALL available cluster resources
        train_data = train_data.coalesce(1)
    
    attempt = 0
    trained_ml_model = None
    while attempt < max_retries:
        try:
            trained_ml_model = model.fit(train_data)
            break
        except Exception as e:
            attempt += 1

            if attempt == max_retries:
                raise RuntimeError(f"train_ml_model(): Training failed after {max_retries} attempts") from e

            print(f"train_ml_model(): [Retry {attempt}/{max_retries}] Training failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)

    return trained_ml_model


def prepare_datasets(
    config: Dict,
    scoring: bool = False,
    baseline: bool = False,
    tft: bool = False,
) -> pyspark.sql.dataframe.DataFrame:
    """
    This function reads from lake train or test dataframe with features

    Args:
        config (Dict): configuration dict
        scoring (bool): if this function is called during scoring procces or modeling procces
        baseline (bool): if this run is for baseline model. in this case we do not have features generated
        tft (bool): if this run is for tft model. in this case we do not have features generated

    Returns:
       pyspark.sql.dataframe.DataFrame: dataframe with features for train or for test
    """
    if scoring:
        if baseline or tft:
            test_data_path = config.params.scoring_part_aggregated_data_path
        else:
            # Upload test data
            save_path = "{0}_{1}_{2}_{3}_{4}{5}".format(
                config.params.pipeline_data_name,
                config.params.aggregation_data_path,
                config.params.features_data_path + "_cannibalization",
                config.params.date_level,
                config.params.train_score_suffix,
                config.params.data_path_res,
            )

            test_data_path = "{0}{1}".format(
                config.params.planning_period_id_path, save_path
            )

        all_available_test_data = spark.read.parquet(
            "{0}{1}".format(adl_connection_str, test_data_path)
        )

        return all_available_test_data

    else:
        if baseline or tft:
            train_data_path = config.params.aggregated_data_path
        else:
            save_path = "{0}_{1}_{2}_{3}_{4}{5}".format(
                config.params.pipeline_data_name,
                config.params.aggregation_data_path,
                config.params.features_data_path,
                config.params.feature_selection_path,
                config.params.train_score_suffix,
                config.params.data_path_res,
            )

            train_data_path = "{0}{1}".format(
                config.params.train_planning_period_id_path, save_path
            )

        all_available_data_train = spark.read.parquet(
            "{0}{1}".format(adl_connection_str, train_data_path)
        )

        return all_available_data_train


def get_transforming_pipeline(
    config: Dict,
    df: pyspark.sql.dataframe.DataFrame,
    encoding: str = Encoding.LABEL_ENCODING
) -> Pipeline:
    """
    This function reads from lake train or test dataframe with features

    Args:
        config (Dict): configuration dict
        df (pyspark.sql.dataframe.DataFrame): if this function is called during scoring procces or modeling procces
        encoding (Encoding): encoding method for categorical columns
    Returns:
       Pipeline: transforming pipeline with StringIndexer and VectorAssembler
    """
    (
        old_categorical_column_names,
        numerical_features,
        new_categorical_column_names,
        enc_categorical_column_names,
    ) = get_columns_info_for_model_training(config, df)

    # Convert categorical columns to numerical due to VectorAssembler input types limitation:
    # https://spark.apache.org/docs/3.5.0/ml-features.html#vectorassembler
    category_indexer = StringIndexer(
        inputCols=old_categorical_column_names,
        outputCols=new_categorical_column_names,
        handleInvalid="keep",
    )

    pipeline_stages = [category_indexer]

    if encoding == Encoding.LABEL_ENCODING:
        feature_assembler = VectorAssembler(
            inputCols=new_categorical_column_names + numerical_features,
            outputCol="features",
            handleInvalid="keep",
        )

        pipeline_stages.extend([feature_assembler])

    elif encoding == Encoding.ONE_HOT_ENCODING:
        # To-Do in future US other encodings will be tested and maybe better one chosen
        encoder = OneHotEncoder(
            inputCols=new_categorical_column_names,
            outputCols=enc_categorical_column_names,
            handleInvalid="keep",
        )

        feature_assembler = VectorAssembler(
            inputCols=enc_categorical_column_names + numerical_features,
            outputCol="features",
            handleInvalid="keep",
        )

        pipeline_stages.extend([encoder, feature_assembler])
    else:
        raise ValueError(F"get_transforming_pipeline(): Unknown encoding method: {encoding}!")

    transforming_pipeline = Pipeline(stages=pipeline_stages)
        
    return transforming_pipeline


def get_hyperparams_for_spark_model(
    model_name: str, input_hyperparams_dict: dict
) -> dict:
    """
    This function reads from lake train or test dataframe with features

    Args:
        model_name (str): name of the model
        input_hyperparams_dict (dict): python dictionary with tuned hyperparams (for lgbm we need to rename keys)

    Returns:
       dict: dictionary with hyperparameters for model with model_name
    """
    if model_name == "lgbm":
        hyperparams_for_spark_model = {
            "learningRate": input_hyperparams_dict["learning_rate"],
            "maxBin": input_hyperparams_dict["max_bin"],
            "numIterations": input_hyperparams_dict["n_estimators"],
            "lambdaL1": input_hyperparams_dict["lambda_l1"],
            "lambdaL2": input_hyperparams_dict["lambda_l2"],
            "numLeaves": input_hyperparams_dict["num_leaves"],
            "featureFraction": input_hyperparams_dict["feature_fraction"],
            "baggingFraction": input_hyperparams_dict["bagging_fraction"],
            "baggingFreq": input_hyperparams_dict["bagging_freq"],
            "minDataInLeaf": input_hyperparams_dict["min_child_samples"],
        }
    elif model_name == "xgboost":
        hyperparams_for_spark_model = input_hyperparams_dict
    else:
        hyperparams_for_spark_model = dict()

    return hyperparams_for_spark_model


def tune_hyperparams_objective_function(
    hyperparams: dict,
    config: Dict,
    model_name: str,
    df: pyspark.sql.dataframe.DataFrame,
    train_data: pyspark.sql.dataframe.DataFrame,
    test_data: pyspark.sql.dataframe.DataFrame,
) -> dict:
    """
    This is main function to run evaluation for one set of hyperparameters inside of tuning step

    Args:
        hyperparams (str): name of the model
        config (Dict): configuration dict
        model_name (str): name of the model
        df (pyspark.sql.dataframe.DataFrame): original train dataframe with features
        train_data (pyspark.sql.dataframe.DataFrame): dataframe for train dates, transformed(indexer + vectorAssembler)
        test_data (pyspark.sql.dataframe.DataFrame): dataframe for test dates, transformed(indexer + vectorAssembler)

    Returns:
       dict: dictionary with metric and status of evaluation ser
    """
    model = get_ml_model(config, model_name, hyperparams, df, shap=False)
    trained_model = train_ml_model(model, train_data)

    forecast = trained_model.transform(test_data)

    evaluator = MAPEEvaluator(
        labelCol=config.params.target_name_cleaned,
        predictionCol=config.params.forecast_column,
    )

    mape = evaluator.evaluate(forecast)

    return {"loss": mape, "status": STATUS_OK}


def get_hyperparams_search_space_for_model(model_name: str) -> dict:
    """
    This is fucnction for setting hyperparametes space for tuning models

    Args:
        model_name (str): name of the model

    Returns:
       dict: dictionary with hyperpot spaces for tuning params of the model
    """
    hyperparams_search_space = {}

    shared_hyperparams_search_space_for_gbt_models = {
        "learning_rate": hp.uniform("learning_rate", 0.001, 0.05),
        "max_bin": hp.randint("max_bin", 8, 1024),
        "n_estimators": hp.randint("n_estimators", 200, 250),
        "verbosity": 3,  # mode - DEBUG
    }

    if model_name == "lgbm":
        hyperparams_search_space = {
            **shared_hyperparams_search_space_for_gbt_models,
            # "objective": "regression",
            # "metric": "regression",
            "lambda_l1": hp.uniform("lambda_l1", 1e-8, 10.0),
            "lambda_l2": hp.uniform("lambda_l2", 1e-8, 10.0),
            "num_leaves": hp.randint("num_leaves", 8, 512),
            "feature_fraction": hp.uniform("feature_fraction", 0.1, 1.0),
            "bagging_fraction": hp.uniform("bagging_fraction", 0.1, 1.0),
            "bagging_freq": hp.randint("bagging_freq", 1, 10),
            "min_child_samples": hp.randint("min_child_samples", 5, 100),
        }
    elif model_name == "xgboost":
        hyperparams_search_space = {
            **shared_hyperparams_search_space_for_gbt_models,
            # "objective": "reg:squarederror",
            # "eval_metric": "rmse",
            # "tree_method": "hist",
            "reg_alpha": hp.uniform("reg_alpha", 1e-8, 10.0),
            "reg_lambda": hp.uniform("reg_lambda", 1e-8, 10.0),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.1, 1.0),
            "subsample": hp.uniform("subsample", 0.1, 1.0),
        }
    elif model_name == "tft":
        pass
    else:
        Exception(
            "Unknown model type to get hyperparams search space for, got {model_name} model!"
        )

    return hyperparams_search_space

from hyperopt import Trials

def tune_hyperparams(
    model_name: str,
    train_data: pyspark.sql.dataframe.DataFrame,
    test_data: pyspark.sql.dataframe.DataFrame,
    config: Dict,
    df: pyspark.sql.dataframe.DataFrame,
) -> dict:
    """
    This fucnctio perform hyperparameter tuning. It creates hyperparam space for each model,
    set function to minimize, run N trials and choose the best one. Hyperopt is used

    Args:
        model_name (str): name of the model
        train_data (pyspark.sql.dataframe.DataFrame): dataframe for train dates, transformed(indexer + vectorAssembler)
        test_data (pyspark.sql.dataframe.DataFrame): dataframe for test dates, transformed(indexer + vectorAssembler)
        config (Dict): configuration dict
        df (pyspark.sql.dataframe.DataFrame): original train dataframe with features
    Returns:
       dict: dictionary best hyperparameter ser
    """

    # Set needed constants for hyperparams tuning process with hyperopt library
    # Needs to be tested with different values
    HYPERPARAMS_SEARCH_ALGORITHM = tpe.suggest  # or rand.suggest
    HYPERPARAMS_TOTAL_MAX_TRIALS_COUNT = 30
    HYPERPARAMS_MAX_TRIALS_COUNT_IN_QUEUE = 2
    search_space = get_hyperparams_search_space_for_model(model_name)
    best_params_prev = {"bagging_fraction": 0.650973747958719, "bagging_freq": 8, "feature_fraction": 0.8721767678897901, "lambda_l1": 2.036094247166061, "lambda_l2": 2.087863022576826, "learning_rate": 0.04507689497382924, "max_bin": 202, "min_child_samples": 80, "n_estimators": 245, "num_leaves": 149}
    trials = Trials()
    # initial_trial = {
    #     'tid': 0,
    #     'state': 2,  # means 'new'
    #     'result': {'status': 'new'},
    #     'misc': {
    #         'tid': 0,
    #         'vals': {
    #             k: [best_params_prev[k]] if k in best_params_prev else [None]
    #             for k in search_space.keys()
    #         }
    #     },
    #     'spec': None,
    #     'owner': None,
    #     'version': 0,
    #     'book_time': None,
    #     'exp_key':None,
    #     'refresh_time': None
    # }
    # trials.insert_trial_docs([initial_trial])
    print('trials', trials )
    # print('trial', trials[0])
    trials.refresh()
    
    fmin_objective = partial(
        tune_hyperparams_objective_function,
        config=config,
        model_name=model_name,
        df=df,
        train_data=train_data,
        test_data=test_data,
    )
    # Documentation about fmin function: https://github.com/hyperopt/hyperopt/wiki/FMin
    best_hyperparams = fmin(
        fn=fmin_objective,
        space=search_space,
        algo=HYPERPARAMS_SEARCH_ALGORITHM,
        max_evals=HYPERPARAMS_TOTAL_MAX_TRIALS_COUNT,
        max_queue_len=HYPERPARAMS_MAX_TRIALS_COUNT_IN_QUEUE,
        show_progressbar=True,
        # trials=trials,
        verbose=True,
    )

    # Evaluate because of returned index rather than value in hp.choice for search space
    # Check: https://stackoverflow.com/questions/45674652/best-parameters-solved-by-hyperopt-is-unsuitable
    best_hyperparams_space_evaluated = space_eval(search_space, best_hyperparams)

    return best_hyperparams_space_evaluated


def upload_json(input_dict: dict, path: str) -> None:
    """Function for uploading dictionary to data lake

    Args:
        input_dict (dict): dictonary (e.g. hyperparams)
        path (str): path to data lake (abfss:/[file_system]@[account_name].dfs.core.windows.net/[path])

    Returns:
        None
    """
    json_object = json.dumps(ast.literal_eval(str(input_dict)))
    dbutils.fs.put(file=path, contents=str(json_object), overwrite=True)


def load_latest_model_mlflow_run(
    config: Dict, mlflow_model_name: str
) -> mlflow.entities.run.Run:
    """
    This function loads latest mlflow run (for retrieving model during scoring step)

    Args:
        config (Dict): configuration dict
        mlflow_model_name (str): name of the registered model

    Returns:
        mlflow.entities.run.Run: run in mlflow registry
    """

    model_runs_sorted = mlflow.search_runs(
        experiment_names=["/final_modeling"],
        output_format="list",
        filter_string=f"status = 'FINISHED' and tags.mlflow.runName = '{mlflow_model_name}'",
        order_by=["start_time DESC"],
    )
    actual_model_run = model_runs_sorted[0]
    return actual_model_run


def get_input_features_names(
    transforming_pipeline: Pipeline,
    encoding: Encoding = Encoding.LABEL_ENCODING
) -> List[str]:
    """
    This function loads list of feature names, that were used during model training from transforming pipeline

    Args:
        transforming_pipeline (Pipeline): transforming pipeline with StringIndexer and VectorAssembler): configuration dict
        encoding (Encoding): encoding method for categorical columns
    Returns:
        List[str]:list of feature names, that were used during model training
    """
    if encoding == Encoding.LABEL_ENCODING:
        category_indexer, vector_assembler = transforming_pipeline.getStages()
    elif encoding == Encoding.ONE_HOT_ENCODING:
        category_indexer, encoder, vector_assembler = transforming_pipeline.getStages()
    else:
        raise ValueError("get_input_features_names(): Unknown encoding type!")

    vector_assembler_input_cols = vector_assembler.getInputCols()
    output_input_cols = category_indexer.getInputCols()

    feature_names_list = vector_assembler_input_cols.copy()
    feature_names_list[: len(output_input_cols)] = output_input_cols

    return feature_names_list


# Main function for calculate metrics and prepare structure for log in MLFlow
def compute_metrics(
    df_complete_validation: pyspark.sql.dataframe.DataFrame,
    window: int,
    config: Dict,
) -> Tuple[typing.Dict[str, float], pyspark.sql.dataframe.DataFrame, int]:
    print(f"[LOG] Process {window} window..")
    # [TMP] Rename pred cols
    df_complete_validation = df_complete_validation.withColumnRenamed(
        "prediction", config.params.forecast_column
    )
    # Check and cast type for Finalclean and Forecast
    df_complete_validation = df_complete_validation.withColumn(
        config.params.forecast_column,
        df_complete_validation[config.params.forecast_column].cast(T.DoubleType()),
    )
    df_complete_validation = df_complete_validation.withColumn(
        config.params.target_name,
        df_complete_validation[config.params.target_name].cast(T.DoubleType()),
    )
    # Filter by window
    df_complete_validation = df_complete_validation.filter(f"window = {window}")
    # Compute metrics by pair
    metrics_by_pair = compute_metrics_by_pair(
        df_complete_validation,
        config.params.target_name,
        config.params.forecast_column,
        config.params.product_levels,
        config.params.business_levels,
    )
    metrics_by_pair_total = metrics_by_pair.describe()
    # Compute mml statistics
    metrics_mml = compute_mml_metrics(
        df_complete_validation, config.params.target_name, config.params.forecast_column
    )
    metrics_mml = metrics_mml.withColumnRenamed("mean_squared_error", "MSE")
    metrics_mml = metrics_mml.withColumnRenamed("root_mean_squared_error", "RMSE")
    metrics_mml = metrics_mml.withColumnRenamed("mean_absolute_error", "MAE")
    metrics_mml = metrics_mml.withColumnRenamed("R^2", "R_2")
    wmapes2 = wmape_total(
        df_complete_validation,
        config.params.target_name,
        config.params.forecast_column,
        config.params.product_levels,
        config.params.business_levels,
    )
    wmape_metric_total = wmapes2.select("WMAPE_total").describe()

    # Prepare to log in MLFlow.
    metrics_dict = {}
    # Process metrics by pair
    for metrics_name in [
        "MAPE",
        "MAE",
        "MSE",
        "FB",
        "RES",
        "WMAPE",
        "WMAPE_smart",
    ]:  # , '']:
        # 1-st element, cause of this is mean value from describe
        if metrics_by_pair_total.collect()[1][metrics_name]:
            metrics_dict[metrics_name + "_pair"] = "{:.4f}".format(
                float(metrics_by_pair_total.collect()[1][metrics_name])
            )

    for metrics_name in ["WMAPE_total"]:
        if wmape_metric_total.collect()[1][metrics_name]:
            metrics_dict[metrics_name] = "{:.4f}".format(
                float(wmape_metric_total.collect()[1][metrics_name])
            )

    # Update dict with total-metrics from mml
    for metrics_name in ["MSE", "RMSE", "R_2", "MAE"]:
        if metrics_mml.collect()[0][metrics_name]:
            metrics_dict[metrics_name + "_total"] = "{:.4f}".format(
                float(metrics_mml.collect()[0][metrics_name])
            )

    metrics_dict = {x: float(y) for x, y in metrics_dict.items()}
    return metrics_dict, metrics_by_pair, df_complete_validation.count()


def impute_missing_keys(input_dict: dict, final_length: int) -> dict:
    """
    This function is for dealing with feature importance values for pyspark models.
    They return feature importance as sparce vector,
    so for correct mapping with feature names we should impute this dictionary

    Args:
        input_dict (dict): input dictionary with not full sequence of keys
        final_length (int): final desired length of dictionary
    Returns:
        dict: dictionary with imputed sequence of keys
    """
    # Get all existing keys
    existing_keys = set(input_dict.keys())

    # Create all possible keys in the range
    all_keys = {f"f{i}" for i in range(final_length)}

    # Find the missing keys
    missing_keys = all_keys - existing_keys

    # Impute the missing keys with value 0
    for key in missing_keys:
        input_dict[key] = 0.0

    # Sort the dictionary by key
    sorted_dict = {
        key: input_dict[key]
        for key in sorted(input_dict.keys(), key=lambda x: int(x[1:]))
    }

    return sorted_dict


def group_and_sum_keys(input_dict: dict) -> dict:
    """
    Function for dealing with feature importance results after one hot encoding.
    One categorical column generate N column with separate importance, so for final result we need to sum that importances

    Args:
        input_dict (dict): input dictionary with importance for each column
    Returns:
        dict: dictionary with imputed sequence of keys
    """
    grouped_dict = defaultdict(float)

    for key, value in input_dict.items():
        # Identify the base key by removing the trailing digits and underscore
        base_key = re.sub(r"_\d+$", "", key)

        # Sum the values for similar keys
        grouped_dict[base_key] += value

    return dict(grouped_dict)


def get_ohe_feature_splitted_names(
    enc_categorical_columns: list,
    train_data: pyspark.sql.dataframe.DataFrame
) -> list:
    """
    Get names of splitted OHE feature columns.
    
    Args:
        enc_categorical_columns: List of encoded categorical column names
        train_data: Training dataset to get unique value counts
    
    Returns:
        list: Splitted one-hot encoded column names
    """
    ohe_columns = []
    
    for cat_col in enc_categorical_columns:
        original_col = cat_col.split("_indexed")[0]
        n_unique = train_data.select(original_col).distinct().count()
        
        # Generate column names for each unique value
        cat_ohe_names = [f"{cat_col}_{i}" for i in range(n_unique + 1)]
        ohe_columns.extend(cat_ohe_names)
    
    return ohe_columns


def get_feature_importance_names(
    config: Dict,
    train_data: pyspark.sql.dataframe.DataFrame,
    encoding: Encoding
):
    """
    Retrieves the feature names used for model training, considering the specified encoding method.

    Args:
        config (Dict): Configuration dictionary containing model parameters.
        train_data (pyspark.sql.dataframe.DataFrame): Training dataset with features.
        encoding (Encoding): Encoding method for categorical columns.

    Returns:
        List[str]: List of feature names used for model training.

    Raises:
        ValueError: If an unknown encoding method is provided.
    """
    (
        old_cat_cols,
        num_cols,
        new_cat_cols,
        enc_cat_cols,
    ) = get_columns_info_for_model_training(config, train_data)

    if encoding == Encoding.LABEL_ENCODING:
        return new_cat_cols + num_cols
    elif encoding == Encoding.ONE_HOT_ENCODING:
        return get_ohe_feature_splitted_names(enc_cat_cols, train_data) + num_cols
    else:
        raise ValueError(f"get_encoded_feature_names(): Unknown encoding method: {encoding}!")


def get_xgboost_importance(model: SparkXGBRegressor, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importance from XGBoost model.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance data sorted by importance (descending)
    """
    # Get sparse importance values
    importance_values = model.get_booster().get_score(importance_type="gain")
    
    # Fill missing keys with zeros to match feature_names length
    complete_importance = impute_missing_keys(importance_values, len(feature_names))
    
    # Map feature indices to actual feature names
    feature_importance_map = {
        feature_names[i]: float(complete_importance.get(f"f{i}", 0.0))
        for i in range(len(feature_names))
    }
    
    importance_df = pd.DataFrame(
        list(feature_importance_map.items()),
        columns=['feature', 'importance']
    )

    importance_df.sort_values(by='importance', ascending=False, inplace=True)
    
    return importance_df


def get_lgbm_importance(model: LightGBMRegressor, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importance from LightGBM model.
    
    Args:
        model: Trained LightGBM model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance data sorted by importance (descending)
    """
    feature_importances = model.getFeatureImportances()
    
    # Create Series and sort by importance
    importance_series = pd.Series(feature_importances, index=feature_names)
    importance_series = importance_series.sort_values(ascending=False)
    
    importance_df = pd.DataFrame({
        'feature': importance_series.index,
        'importance': importance_series.values
    })
    
    return importance_df


def get_model_importance(
    config: Dict, model_name: str, model, train_data: pyspark.sql.dataframe.DataFrame
) -> pd.DataFrame:
    """
    Function for calculating feature importance for the model

    Args:
        config (Dict): configurating dictionary
        model_name (str): model name
        model (Union[LightGBMRegressor, xgboost.spark.estimator.SparkXGBRegressor]): trained model object
        train_data (pyspark.sql.dataframe.DataFrame): dataframe with all columns, that was used for training

    Returns:
        pd.DataFrame: dataframe with one column for feature name and other for its importance

    Raises:
        ValueError: If an usupproted model is provided.
    """
    if model_name not in ['xgboost', 'lgbm']:
        raise ValueError(f"get_model_importance(): Unsupported model: {model_name}!")
    
    encoding = config.params.model_encodings[model_name]
    feature_names = get_feature_importance_names(config, train_data, encoding)
    
    if model_name == "xgboost":
        return get_xgboost_importance(model, feature_names)
    elif model_name == "lgbm":
        return get_lgbm_importance(model, feature_names)


def get_function_mappings() -> dict[str, Callable]:
    """
    Function to get a mapping of functions from function names

    Args:

    Returns:
        dict[str, Callable]: dict with name to function mapping
    """
    function_mapping = {
        "sum": F.sum,
        "mean": F.mean,
        "std": F.stddev,
        "last": F.last,
        "min": F.min,
        "count": F.count,
        "countDistinct": F.countDistinct,
        "max": F.max,
    }

    return function_mapping


def calculate_minimal_lag_value_for_possible_datetime_levels(
    config: Dict,
) -> typing.Dict[str, int]:
    """
    Function to calculate minimal lag value to take into account for each possible datetime level.
    Used in CANNIBALIZATION VBB's to correctly calculate datetime level's offset for feature containing target values

    Args:
        config (Dict): config object

    Returns:
        typing.Dict[str, int]: dictionary with calculated minimal lag value for each datetime level
    """

    current_datetime_forecasting_level = config.params.date_level

    limb_period_for_current_datetime_forecasting_level = config.params.limb_period

    minimal_lag_values_dict = dict()

    for possible_datetime_level in config.params.datetime_levels:
        if possible_datetime_level == current_datetime_forecasting_level:
            minimal_lag_values_dict[
                possible_datetime_level
            ] = limb_period_for_current_datetime_forecasting_level
        else:
            if (
                current_datetime_forecasting_level == "Day"
                and possible_datetime_level == "Week"
            ):
                minimal_lag_values_dict[possible_datetime_level] = math.ceil(
                    limb_period_for_current_datetime_forecasting_level / 7
                )
            if (
                current_datetime_forecasting_level == "Day"
                and possible_datetime_level == "Month"
            ):
                minimal_lag_values_dict[possible_datetime_level] = math.ceil(
                    limb_period_for_current_datetime_forecasting_level / 28
                )
            if (
                current_datetime_forecasting_level == "Day"
                and possible_datetime_level == "Year"
            ):
                minimal_lag_values_dict[possible_datetime_level] = math.ceil(
                    limb_period_for_current_datetime_forecasting_level / 365
                )

            if (
                current_datetime_forecasting_level == "Week"
                and possible_datetime_level == "Month"
            ):
                minimal_lag_values_dict[possible_datetime_level] = math.ceil(
                    limb_period_for_current_datetime_forecasting_level / 4
                )
            if (
                current_datetime_forecasting_level == "Week"
                and possible_datetime_level == "Year"
            ):
                minimal_lag_values_dict[possible_datetime_level] = math.ceil(
                    limb_period_for_current_datetime_forecasting_level / 52
                )

            if (
                current_datetime_forecasting_level == "Month"
                and possible_datetime_level == "Year"
            ):
                minimal_lag_values_dict[possible_datetime_level] = math.ceil(
                    limb_period_for_current_datetime_forecasting_level / 12
                )

    return minimal_lag_values_dict


def get_date_feature_expression_to_calculate(
    feature_name_literal: str, config: Dict
) -> SparkColumn:
    """
    Function to calculate pyspark's column date expression based on passed input feature name's string literal.

    Args:
        feature_name_literal (str): string literal of column to calculate
        config (Dict): config object

    Returns:
        SparkColumn: pyspark's calculated column for date expression
    """

    match feature_name_literal:
        case "Year":
            feature_expression = F.year(config.params.date_column)
        case "Quarter":
            feature_expression = F.quarter(config.params.date_column)
        case "Month":
            feature_expression = F.month(config.params.date_column)
        case "WeekOfYear":
            feature_expression = F.weekofyear(config.params.date_column)
        case "Week":
            week_first_date_dict = {
                "Monday": "2009-12-28",
                "Tuesday": "2009-12-29",
                "Wednesday": "2009-12-30",
                "Thursday": "2009-12-31",
                "Friday": "2010-01-01",
                "Saturday": "2010-01-02",
                "Sunday": "2010-01-03",
            }

            start_date = F.lit(week_first_date_dict[config.params.week_start_day])
            end_date = F.col(config.params.date_column)

            feature_expression = F.floor(F.date_diff(end_date, start_date) / 7)
        case "DayOfWeek":
            feature_expression = F.dayofweek(config.params.date_column)
        case "DayOfMonth":
            feature_expression = F.dayofmonth(config.params.date_column)
        case "DayOfYear":
            feature_expression = F.dayofyear(config.params.date_column)
        case other:
            raise Exception(
                f"Unknown date feature name to generate feature expression: {feature_name_literal}"
            )

    return feature_expression.alias(feature_name_literal)


def get_aggregation_levels_from_base_feature(
    base_feature: str, config: Dict
) -> Tuple[str, str, str]:
    """
    Function to extract product, business and datetime levels from aggregation feature name.
    Used in calculating aggregation lag, moving and expanding features

    Args:
        base_feature (str): base aggregation feature name
        config (Dict): config object

    Returns:
         Tuple[str, str, str]: tuple with extracted product, business and datetime levels from 'base_feature'
    """

    pivot_features_config = config.features_config["CANNIBALIZATION"]["pivot_features"]
    aggregation_features_config = config.features_config["CANNIBALIZATION"][
        "aggregation_features"
    ]
    # additional checks if some features are not selected, so we won't have TypeError if expanding 
    # feature should be based on aggregation feature, but there are no pivot features selected and vice versa
    if aggregation_features_config == {}:
        aggregation_features_config = []
    if pivot_features_config == {}:
        pivot_features_config = []

    feature_config = list(
        filter(
            lambda feature_config: feature_config["feature_name"] == base_feature,
            pivot_features_config + aggregation_features_config,
        )
    )[0]

    feature_inputs_dict = feature_config["feature_inputs"]

    product_aggregation_level = feature_inputs_dict["product_level"]
    business_aggregation_level = feature_inputs_dict["business_level"]
    datetime_aggregation_level = feature_inputs_dict["datetime_level"]

    return (
        product_aggregation_level,
        business_aggregation_level,
        datetime_aggregation_level,
    )


def generate_aggregated_date_column(old_date_column: str, config: Dict) -> SparkColumn:
    """
    Function to generate pyspark's column aggregated expression based on passed input date column's string literal.

    Args:
        old_date_column (str): string literal of date column to calculate aggregated expression for
        config (Dict): config object

    Returns:
        SparkColumn: pyspark's generated column for aggregated date expression
    """

    if config.params.date_level == "Day":

        aggregated_date_column_name_expr = F.col(old_date_column)

    elif config.params.date_level == "Week":
        weekday_numbers_dict = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6,
        }

        week_days_diff_expresion = (
            F.weekday(old_date_column)
            - weekday_numbers_dict[config.params.week_start_day]
        )

        days_to_subtract_expression = F.when(
            week_days_diff_expresion >= 0, week_days_diff_expresion
        ).otherwise(week_days_diff_expresion + 7)

        aggregated_date_column_name_expr = F.date_sub(
            old_date_column, days_to_subtract_expression
        )

    elif config.params.date_level in ["Month", "Year"]:

        aggregated_date_column_name_expr = F.trunc(
            old_date_column, config.params.date_level.lower()
        )

    else:
        raise Exception(f"Unknown date aggregation level: {config.params.date_level}")

    return aggregated_date_column_name_expr


# Use regex to extract the base names (everything before the last underscore and digit)
def get_base_name(col_name: str) -> str:
    """Function for subtracting  the last underscore and digit from the column name

    Args:
        col_name (str): column name

    Returns:
        str: base name
    """
    return re.sub(r"_\d+$", "", col_name)


def group_columns_shap(
    df_transformed: pyspark.sql.dataframe.DataFrame,
) -> pyspark.sql.dataframe.DataFrame:
    """Function for grouping columns by base name (for ohe encoding)

    Args:
        df_transformed (pyspark.sql.dataframe.DataFrame): input dataset with SHAP values

    Returns:
        pyspark.sql.dataframe.DataFrame: dataset with grouped and summed columns based on the base name (for ohe encoding)
    """

    # Assuming your column names are in this format: 'Region_indexed_encoded_shap_0', etc.
    columns = df_transformed.columns

    # Create a dictionary where keys are base names and values are lists of columns that share the base name
    base_column_groups = {}
    for col in columns:
        base_name = get_base_name(col)
        if base_name not in base_column_groups:
            base_column_groups[base_name] = []
        base_column_groups[base_name].append(col)
    # Sum columns that share the same base name
    # Create new columns by summing up the original ones
    for base_name, col_group in base_column_groups.items():

        if (
            len(col_group) > 1
        ):  # Only sum if there are multiple columns with the same base name
            df_transformed = df_transformed.withColumn(
                base_name, sum(F.col(c) for c in col_group)
            )
        else:
            df_transformed = df_transformed.withColumn(base_name, F.col(col_group[0]))

    # Drop the original columns that were combined
    cols_to_drop = [
        col for cols in base_column_groups.values() for col in cols if len(cols) > 1
    ]
    df_transformed = df_transformed.drop(*cols_to_drop)

    return df_transformed


def get_shap_column_names(
    config: Dict, mlflow_model_run: mlflow.entities.run.Run, model_name: str
) -> List[str]:
    """Function for getting the SHAP column names for each model, considering encoding

    Args:
        config (Dict): configuration dictionary
        mlflow_model_run (mlflow.entities.run.Run): mlflow model run
        model_name (str): model name

    Returns:
        List(str): list of SHAP column names
    
    Raises:
        ValueError: If an unsupported model is provided.
    """

    if model_name not in ['xgboost', 'lgbm']:
        raise ValueError(f"get_shap_column_names(): Unsupported model: {model_name}!")

    encoding = config.params.model_encodings[model_name]

    if encoding == Encoding.LABEL_ENCODING:
        # Optimization to not read train data
        features_list = ast.literal_eval(mlflow_model_run.data.tags["features list"])
        features_with_shap = [column_name + "_shap" for column_name in features_list] + [
            "Base_value"
        ]
        return features_with_shap
    
    elif encoding == Encoding.ONE_HOT_ENCODING:
        # Prepare train data
        config_train_path = config.params.train_planning_period_id_path + "config.json"
        config_train = get_config_from_adl(config_train_path)

        train_data = prepare_datasets(config_train, baseline=False)
        train_data = train_data.drop(*config.params.fs.feature_to_drop_from_modeling)

        # Get feature importance names
        feature_names = get_feature_importance_names(config, train_data, encoding)

        # Add '_shap' or '_shap_' part into feature names
        is_ohe_splitted_col = lambda s: bool(re.search(r'_\d+$', s))
        insert_shap = lambda s: re.sub(r'_(\d+)$', r'_shap_\1', s)

        features_with_shap = []

        for feature in feature_names:
            if is_ohe_splitted_col(feature):
                features_with_shap.append(insert_shap(feature))
            else:
                features_with_shap.append(feature + "_shap")
        
        return features_with_shap + ["Base_value"]
    else:
        raise ValueError(f"get_shap_column_names(): Unsupported encoding method: {encoding}!")


def get_date_relative_delta(date_level: str, delta: int) -> relativedelta:
    """
    Returns a relativedelta object based on the specified date level and delta value.

    Args:
        date_level (str): level of date aggregation. Can be "Day", "Week", "Month", or "Year"
        delta (int): number of units to add or subtract

    Returns:
        relativedelta: relativedelta object representing the specified delta.

    Raises:
        ValueError: if the date_level is not one of the expected values.
    """
    match date_level:
        case "Day":
            relative_delta = relativedelta(days=delta)
        case "Week":
            relative_delta = relativedelta(weeks=delta)
        case "Month":
            relative_delta = relativedelta(months=delta)
        case "Year":
            relative_delta = relativedelta(years=delta)
        case _:
            raise ValueError(f"Unknown date level: {date_level}!")

    return relative_delta


def get_tft_models(data: DataFrame, config: Dict) -> DataFrame:
    """
    Function to create models dataframe for each store-product combination

    Args:
        data (DataFrame): data to train models
        config (Dict): config dictionary

    Returns:
        DataFrame: dataframe with models for each pair
    """

    tft_model_schema = T.StructType(
        [
            T.StructField(config.params.business_level, T.StringType()),
            T.StructField(config.params.product_level, T.StringType()),
            T.StructField("model", T.StringType(), False),
        ]
    )

    @pandas_udf(tft_model_schema, PandasUDFType.GROUPED_MAP)
    def train_models(history_pd: pd.DataFrame) -> pd.DataFrame:
        """
        Pandas udf function to train models for each store-product combination

        Args:
            history_pd (pd.DataFrame): history data for a store-product combination

        Returns:
            pd.DataFrame: model for each pair
        """
        # if history is too short to forecast (only 1 time-entry)
        if history_pd.shape[0] == 1:
            offset_from_prev_date = get_date_relative_delta(
                date_level=config.params.date_level, delta=1
            )

            # Duplicate row to make enough data for forecasting
            history_pd = pd_concat([history_pd, history_pd], ignore_index=True)

            history_pd.at[0, "ds"] = history_pd.at[0, "ds"] - offset_from_prev_date

        # instantiate the model, configure the parameters
        model = Prophet(
            interval_width=0.95,
            growth="linear",
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            seasonality_mode="multiplicative",
        )

        try:
            # fit the model
            model.fit(history_pd)
            model_json = model_to_json(model)
        except:
            model_json = ""

        history_pd["model"] = model_json

        return history_pd[
            [config.params.business_level, config.params.product_level, "model"]
        ]

    results = (
        data.select(
            config.params.business_level,
            config.params.product_level,
            F.col(config.params.date_column).alias("ds"),
            F.col(config.params.target_name_cleaned).alias("y"),
        )
        .groupBy(config.params.business_level, config.params.product_level)
        .apply(train_models)
        .distinct()
    )

    return results


def get_tft_forecasts(
    test_data: DataFrame, models: DataFrame, config: Dict
) -> DataFrame:
    """
    Function to score models for each store-product combination

    Args:
        test_data (DataFrame): test data for a store-product combination
        models (DataFrame): models for each store-product combination
        config (Dict): config dictionary

    Returns:
        DataFrame: scored data for each store-product combination
    """
    tft_prediction_schema = T.StructType(
        [
            T.StructField(config.params.business_level, T.StringType()),
            T.StructField(config.params.product_level, T.StringType()),
            T.StructField("ds", T.DateType()),
            T.StructField("yhat", T.DoubleType()),
            T.StructField("model", T.StringType(), False),
        ]
    )

    @pandas_udf(tft_prediction_schema, PandasUDFType.GROUPED_MAP)
    def score_models(test_pd: pd.DataFrame) -> pd.DataFrame:
        """
        Pandas udf function to score models for each store-product combination

        Args:
            test_pd (pd.DataFrame): test data for a store-product combination

        Returns:
            pd.DataFrame: forecasts for each pair
        """
        model_json = test_pd["model"].iloc[0]

        if model_json:
            model = model_from_json(model_json)
            prediction = model.predict(test_pd[["ds"]])["yhat"]

            test_pd["yhat"] = prediction.values
        else:
            test_pd["yhat"] = np.NaN

        return test_pd

    test_data = test_data.select(
        config.params.business_level,
        config.params.product_level,
        F.col(config.params.date_column).alias("ds"),
    ).dropDuplicates()  # To drop duplicate rows with different PromoCounter values

    test_data = test_data.join(
        models,
        how="left",
        on=[config.params.business_level, config.params.product_level],
    ).fillna("", subset=["model"])
    results = (
        test_data.groupBy(config.params.business_level, config.params.product_level)
        .apply(score_models)
        .withColumnRenamed("yhat", "Forecast")
        .withColumnRenamed("ds", config.params.date_column)
        .drop("model")
    )

    impute_value = (
        results.select("Forecast")
        .where("Forecast is not null")
        .select(F.avg("Forecast"))
        .collect()[0][0]
    )
    results = results.fillna(impute_value, subset=["Forecast"])

    return results


def get_all_features_from_feature_constructor(
    features_config: Dict,
) -> List[str]:
    """
    Function to score models for each store-product combination

    Args:
        features_config: features config from config object

    Returns:
        List[str]: list of all feature names from 'features_config'
    """
    all_features_from_config = [
        feature_config["feature_name"]
        for feature_group_configs in features_config.values()
        for feature_configs in feature_group_configs.values()
        for feature_config in feature_configs
    ]

    return all_features_from_config
