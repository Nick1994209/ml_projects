from typing import Union, List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns


def to_categorical(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for column in columns:
        df = pd.concat([
            df.drop(column, axis=1), 
            pd.get_dummies(df[column], prefix=column)  # to categorical
        ], axis=1)
    return df


def get_count_categorical(df: pd.DataFrame, columns: Union[List[str], str], sort: bool = False):
    """
    Get amount every category in dataframe
    """
    columns = columns if isinstance(columns, list) else [columns]
    categorical_data = to_categorical(df[columns], columns).astype(bool).sum(axis=0)
    if sort:
        return categorical_data.sort_values(ascending=False)
    return categorical_data


def is_categorical_data(dataframe, count_unique_values):
    return len(dataframe.unique()) <= count_unique_values


def draw_categorical_violin(df: pd.DataFrame, categorical_column: str, target_column: str, with_nan=True,
                            with_count_elements=True) -> None:
    """
    Draw sns.violinplot for every value in categorical_column
    
    with_nan=True: categorical translate to str for catching nan
    with_count_elements: draw count categorical_values under label name
    """
    categorical_values = df[categorical_column].map(str) if with_nan else df[categorical_column]
    categorical_values = pd.get_dummies(categorical_values)
    
    for row_index, row in categorical_values.iterrows():
        for column, exists in row.items():
            if exists:
                categorical_values.loc[row_index, column] = df.loc[row_index, target_column]
            else:
                categorical_values.loc[row_index, column] = None
    
    ax = sns.violinplot(data=categorical_values, palette="Set3", cut=1, linewidth=0.5)
    if with_count_elements:
        count_values = categorical_values.count()
        labes_names = ['%s\n%d'% (c, count_values[c]) for c in categorical_values.columns]
        ax.set_xticklabels(labes_names)
    
    plt.xlabel(categorical_column)
    plt.ylabel(target_column)
    plt.tight_layout()  # Automatically adjust subplot parameters to give specified padding.


def draw_relation_columns_to_target(df: pd.DataFrame, target_column: str, columns: List[str],
                                    figsize=(16, 36), count_unique_values_in_categorical=10) -> None:
    """
    Draw relation columns to target: scatter
    and draw boxplot column
    
    columns: if not defined => by default get all dataframe columns
    
    count_unique_values_in_categorical: if unique values in column <= this value 
        => draw draw_categorical_violin
    
    WARNING: draw_relation_columns_target is so slow function!
    ADVICE:
        example_ussage:
            all_columns = list(set(df.columns) - {target_column})
            draw_relation_columns_to_target(df, target_column, columns=all_columns[:10])
    """
    plt.figure(num=len(columns), figsize=figsize)
    target_values = df[target_column]
    
    count_chart_columns = 2
    count_chat_rows = len(columns)
    for index, column in enumerate(columns):
        str_column_values = df[column].map(str)
        
        if is_categorical_data(str_column_values, count_unique_values_in_categorical):
            plt.subplot(count_chat_rows, count_chart_columns, index*2 + 1)
            draw_categorical_violin(df, column, target_column)
            continue
        
        column_type = df[column].dtype.name
        if column_type in ('int64', 'float64'):
            plt.subplot(count_chat_rows, count_chart_columns, index*2 + 1)
            sns.scatterplot(target_values, df[column])
        
            plt.subplot(count_chat_rows, count_chart_columns, index*2 + 2)
            sns.boxplot(df[column]);
        elif column_type == 'object':
            plt.subplot(count_chat_rows, count_chart_columns, index*2 + 1)
            sns.scatterplot(target_values, str_column_values)
        else:
            # if this exception was raised => add column_type to 'else column_type not in ('object', your_type)
            raise Exception('I do not know about type=%s' % column_type)

    plt.show()
    

def vizualize_categorical(df: pd.DataFrame, column: str, target: str) -> None:
#     sns.catplot(x=column, y=target, data=df);
    sns.relplot(x=column, y=target, data=df);
    return get_count_categorical(df, column, sort=True)
    

def fill_nan_mean(df: pd.DataFrame, columns: List[str]) -> None:
    df = df.copy()
    for column in columns:
        df[column] = df[column].fillna((df[column].mean()))    
    return df
    

def get_count_nan(df: pd.DataFrame, columns: Union[List[str], str]) -> None:
    if isinstance(columns, str):
        columns = [columns]
    
    return {
        column: df[column].isna().sum()
        for column in columns
    }
#     s = df[column].isna().sum()
#     assert df[column].isna().sum() == 0, 'count not nan values %s' % s


def series_to_float(array: pd.Series) -> np.ndarray:
    return array.map(float).values


def get_abroad_values(array: np.ndarray, tresh_hold: int = 4) -> np.ndarray:
    """
    Get values that were been outside the standard deviation threshold
    
    array: np.ndarray[float]  # required float
    returned: np.ndarray[bool]
    
    example usage:
        float_area = series_to_float(df['GrLivArea'])
        is_abroad_area = get_abroad_values(float_area)
        df.loc[is_abroad_area, 'GrLivArea']
    """
    is_array = len(array.shape) == 1

    if is_array:
        # is array
        matrix = array.reshape(len(array), 1)  # matrix is required for StandardScaler
        scaled = StandardScaler().fit_transform(matrix)[:, 0]
    else:
        scaled = StandardScaler().fit_transform(array)
    return ((scaled > tresh_hold) | (scaled < -1 * tresh_hold))