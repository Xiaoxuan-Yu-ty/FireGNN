# -*- coding: utf-8 -*-

"""Carry out Radical search to identify extreme samples in the dataset and give them a single sample score."""

from typing import Callable, Optional, List, Tuple, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandas._typing as pdt
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF
from tqdm import tqdm
import pandera.typing as pat


def do_radical_search(
        data: pd.DataFrame,
        design: pd.DataFrame,
        threshold: float,
        control: str,
        control_based: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Identify the samples with extreme feature values either based on the entire dataset or control population.

    :param data: Dataframe containing the gene expression values
    :param design: Dataframe containing the design table for the data
    :param threshold: Threshold for choosing patients that are "extreme" w.r.t. the controls
    :param control: label used for representing the control in the design table of the data
    :param control_based: The scoring is based on the control population instead of entire dataset
    :return: Dataframe containing the Single Sample scores using radical searching
    """
    # Transpose matrix to get the patients as the rows
    data_transpose = data.transpose()

    # Give each label an integer to represent the labels during classification
    label_mapping = {
        key: val
        for val, key in enumerate(np.unique(design['Target']))
    }

    # Make sure the number of rows of transposed data and design are equal
    assert len(data_transpose) == len(design), 'Data doesnt match the design matrix'

    # Create a dataframe initialized with 0's [patients x features]
    output_df = pd.DataFrame(0, index=data_transpose.index, columns=data_transpose.columns)

    # Values that are greater than the threshold or lesser than negative threshold are considered as extremes.
    upper_thresh = 1 - (threshold / 100)
    lower_thresh = (threshold / 100)

    if control_based:
        controls = data_transpose[list(design.Target == control)]

        # Calculate the empirical cdf for every gene and get the cdf score for the data
        control_ecdf = controls.apply(_get_ecdf, step=False, extrapolate=True).values
        cdf_score = _apply_func(data_transpose, control_ecdf).fillna(0)

        # Check if each patient's feature is over or under expressed compared to the control population
        output_df = pd.DataFrame(np.where(cdf_score.values > upper_thresh, 1, output_df.values))
        output_df = pd.DataFrame(np.where(cdf_score.values < lower_thresh, -1, output_df.values))

    else:
        # Calculate the empirical cdf for every gene and get the cdf score for the data
        feature_to_ecdf = {
            feature: _get_ecdf(data_transpose[feature].values)
            for feature in data_transpose
            if len(data_transpose[feature].unique()) > 1  # Check not all values are the same
        }

        # Iterate over patients and check if any of its features is significant
        for patient_index, features in data_transpose.iterrows():

            # Iterate over patient features
            for feature, value in features.items():

                # Skip if feature has no calculated eCDF
                if feature not in feature_to_ecdf:
                    continue

                # Calculate position of the patient in the distribution of the feature
                patient_position_in_distribution = float(feature_to_ecdf[feature]([value])[0])

                if patient_position_in_distribution <= lower_thresh:
                    output_df.loc[patient_index, feature] = -1

                if patient_position_in_distribution > upper_thresh:
                    output_df.loc[patient_index, feature] = 1

    output_df.columns = data.index
    output_df.index = data.columns

    summary_df = output_df.apply(pd.Series.value_counts)

    # Add labels to the data samples
    label = design['Target'].map(label_mapping)
    label.reset_index(drop=True, inplace=True)

    output_df['label'] = label.values

    return output_df, summary_df


def _get_ecdf(
        obs: pdt.ArrayLike,
        side: str = 'right',
        step: bool = True,
        extrapolate: bool = False
) -> Any:
    """Calculate the Empirical CDF of an array and return it as a function.

    :param obs: Observations
    :param side: Defines the shape of the intervals constituting the steps. 'right' correspond to [a, b) intervals
        and 'left' to (a, b]
    :param step: Boolean value to indicate if the returned value must be a step function or an continuous based on
        interpolation or extrapolation function
    :param extrapolate: Boolean value to indicate if the continuous must be based on extrapolation
    :return: Empirical CDF as a function
    """
    if step:
        return ECDF(x=obs, side=side)
    else:
        obs = np.array(obs, copy=True)
        obs.sort()

        num_of_obs = len(obs)

        y = np.linspace(1. / num_of_obs, 1, num_of_obs)

        if extrapolate:
            return interp1d(obs, y, bounds_error=False, fill_value="extrapolate")  # type: ignore
        else:
            return interp1d(obs, y)


def _apply_func(
        df: pd.DataFrame,
        func_list: npt.NDArray[Any]
) -> pd.DataFrame:
    """Apply functions from the list (in order) on the respective column.

    :param df: Data on which the functions need to be applied
    :param func_list: List of functions to be applied
    :return: Dataframe which has been processed
    """
    final_df = pd.DataFrame()

    new_columns = [index for index, _ in enumerate(df.columns)]
    old_columns = list(df.columns)

    df.columns = pd.Index(new_columns)

    for idx, i in enumerate(tqdm(df.columns, desc='Searching for radicals: ')):
        final_df[i] = np.apply_along_axis(func_list[idx], 0, df[i].values)  # type: ignore

    final_df.columns = pd.Index(old_columns)

    return final_df

def do_z_score(
        data: pd.DataFrame,
        design: pd.DataFrame,
        control: str = 'Control',
        threshold: float = 2.0,
) -> pd.DataFrame:
    """Carry out Z-Score based single sample DE analysis.

    :param data: Dataframe containing the gene expression values
    :param design: Dataframe containing the design table for the data
    :param control: label used for representing the control in the design table of the data
    :param threshold: Threshold for choosing patients that are "extreme" w.r.t. the controls.
    :return: Dataframe containing the Single Sample scores using Z_Scores
    """
    # Check if the control variable is as per the R Naming standards
    assert control[0].isalpha(), "Please pass the control indicator contains atleast 1 alphabet."

    # Transpose matrix to get the patients as the rows
    data = data.transpose()

    # Give each label an integer to represent the labels during classification
    label_mapping = {
        key: val
        for val, key in enumerate(np.unique(design['Target']))
    }

    # Make sure the number of rows of transposed data and design are equal
    assert len(data) == len(design)

    # Extract the controls from the dataset
    controls = data[list(design.Target == control)]

    # Calculate the "Z Score" of each individual patient
    mean = controls.mean(axis=0)
    std = controls.std(axis=0)
    z_scores = (data - mean) / std

    out_z_scores = z_scores.copy()

    # Values that are greater than the 2 sigma or lesser than negative 2 sigma are considered as extremes

    out_z_scores[z_scores > threshold] = 1
    out_z_scores[z_scores < -threshold] = -1

    # Values between upper and lower limit are assigned 0
    out_z_scores[(z_scores < threshold) & (z_scores > -threshold)] = 0

    df = pd.DataFrame(data=out_z_scores, index=data.index, columns=data.columns)

    label = design['Target'].map(label_mapping)
    label.reset_index(drop=True, inplace=True)

    output_df = df.apply(_bin).copy()

    output_df['label'] = label.values

    return output_df


def _bin(row: pat.Series[int]) -> List[int]:
    """Replace values greater than 0 as 1 and lesser than 0 as -1."""
    return [
        1 if (val > 0) else (-1 if (val < 0) else 0)
        for val in row
    ]


def do_avg_score(data: pd.DataFrame, design:pd.DataFrame, output_dir):

    data = data.transpose()
    assert len(data) == len(design)

    avg_patient = data.mean(axis=1)
    avg_gene = data.mean(axis=0)

    pat_score = data.copy()
    gene_score = data.copy()

    # score based on patient-level average
    for idx in pat_score.index:
        pat_score.loc[idx,:] = [1 if x>= avg_patient.loc[idx] else 0 for x in pat_score.loc[idx].values]
        
    # score based on gene-leval average
    for col in gene_score.columns:
        gene_score.loc[:, col] = [1 if x>=avg_gene.loc[col] else 0 for x in gene_score[col].values]
    pat_score.to_csv(f'{output_dir}/adni_patient_avg.csv')
    gene_score.to_csv(f'{output_dir}/adni_gene_average.csv')
    
    return pat_score, gene_score 
