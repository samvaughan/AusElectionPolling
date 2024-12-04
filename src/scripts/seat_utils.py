import re
import numpy as np
from scipy.stats import gaussian_kde as kde
import matplotlib.pyplot as plt
import pandas as pd
import scipy.special as sp


## Data Transformations
def centre_categorical_column(column):

    return (column.cat.codes - column.cat.codes.mean()) / column.cat.codes.std()


def weighted_value(df, bin_centre_function, starting_string):
    """Make a single weighted average column from a list of bin fractions.

    Args:
        df (pd.DataFrame): Dataframe with propensity data. Must have columns starting 'age:' or 'income:', etc
        bin_centre_function (func): A function to extract a single numerical value from a column name
        starting_string (_type_): Starting string for columns we want to select

    Returns:
        pd.Series: A series of weighted values
    """

    df = df.loc[:, df.columns.str.startswith(starting_string)]

    column_names = list(df.columns)

    bin_centres = np.array(list(map(bin_centre_function, column_names)))

    mean_values = np.nansum(df * bin_centres, axis=1) / np.nansum(df, axis=1)

    return mean_values


def weighted_age(df):

    mean_ages = weighted_value(df, age_bin_centres, "age")

    return mean_ages


def weighted_income(df):

    mean_ages = weighted_value(df, income_bin_centres, "income")

    return mean_ages


def weighted_education(df):

    mean_education = weighted_value(df, education_bin_centres, "heap")

    return mean_education


def education_bin_centres(column_name):

    education_levels_dict = {
        "heap: Postgraduate Degree Level_pc": 4,
        "heap: Graduate Diploma and Graduate Certificate Level_pc": 4,
        "heap: Bachelor Degree Level_pc": 3,
        "heap: Advanced Diploma and Diploma Level_pc": 3,
        "heap: Certificate III & IV Level_pc": 2,
        "heap: Secondary Education - Years 10 and above_pc": 2,
        "heap: Certificate I & II Level_pc": 1,
        "heap: Secondary Education - Years 9 and below_pc": 1,
        "heap: Supplementary Codes_pc": 0,
        "heap: Not stated_pc": 0,
        "heap: Not applicable_pc": 0,
    }

    return education_levels_dict[column_name]


def age_bin_centres(column_name):

    age_range = re.sub("[^0-9-]", "", column_name).split("-")
    age_range = list(map(int, age_range))

    return np.mean(age_range)


def income_bin_centres(column_name):
    """Turn the colum name into an average income

    An example column: 'income: $500-$649 ($26,000-$33,799)_pc'

    Also make sure that 'Nil Income', 'Not Stated', 'Not Applicable' and 'Negative Income' return 0

    Returns:
        _type_: _description_
    """

    if column_name in [
        "income: Negative income_pc",
        "income: Nil income_pc",
        "income: Not stated_pc",
        "income: Not applicable_pc",
    ]:
        return 0.0

    # Get rid of anything that's not a number, a dash or a bracket
    # For the income, we want to take the yearly value which is the second of the two
    income_range = re.sub("[^0-9-(]", "", column_name).split("(")[1].split("-")

    income_range = list(map(int, income_range))
    return np.mean(income_range)


def education_levels_renamer():
    """A dictionary to combine and rename different education categories

    Returns:
        _type_: _description_
    """

    # Rename the eudcation levels according to the following values:
    educ_renamer = {
        "Postgraduate Degree Level": "Postgrad",
        "Graduate Diploma and Graduate Certificate Level": "Postgrad",
        "Bachelor Degree Level": "Undergrad",
        "Advanced Diploma and Diploma Level": "Undergrad",
        "Certificate III & IV Level": "HighSchool",
        "Certificate I & II Level": "HighSchool",
        "Secondary Education - Years 10 and above": "NoHighSchool",
        "Secondary Education - Years 9 and below": "NoHighSchool",
    }
    return educ_renamer


def plot_density(data, nbins=50, cmap="Greys", **kwargs):

    data = data.dropna()
    k = kde(data.T)

    x = data.iloc[:, 0]
    y = data.iloc[:, 1]

    xi, yi = np.mgrid[x.min() : x.max() : nbins * 1j, y.min() : y.max() : nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    fig, ax = plt.subplots()
    ax.contour(xi, yi, zi.reshape(xi.shape), cmap=cmap, **kwargs)
    ax.set_xlabel(data.columns[0])
    ax.set_ylabel(data.columns[1])

    return fig, ax


# Simulated data
def simulate_sample(N, param_dict):
    """
    Simulate a survey/poll/census with N responses and the coefficients from coeff_dict
    """

    bias = param_dict["bias"]
    rng = param_dict["rng"]
    noise = param_dict["noise"]

    df = pd.DataFrame()
    p_green = bias
    for column_name, dict_values in param_dict["fixed_effects"].items():

        func = dict_values["func"]
        df[column_name] = func(size=N, **dict_values["kwargs"])

        df[column_name] = pd.Categorical(
            df[column_name], categories=dict_values["categories"], ordered=True
        )

        # Now add this to the probability of voting green
        p_green += (
            (df[column_name].cat.codes - df[column_name].cat.codes.mean())
            / df[column_name].cat.codes.std()
        ) * dict_values["coeff"]

    for column_name, dict_values in param_dict["random_effects"].items():

        # This is now a dict, not a single value
        params = dict_values["coeff"]
        func = dict_values["func"]
        df[column_name] = func(size=N, **dict_values["kwargs"])
        # Now add this to the probability of voting green

        p_green += np.array([params[level] for level in df[column_name]])

    # Now centre the survey data
    df["p_green_true"] = sp.expit(p_green)
    df["p_green_noisy"] = sp.expit(p_green + rng.normal(loc=0, scale=noise, size=N))

    df["survey_response"] = rng.binomial(1, df["p_green_noisy"], N)

    return df


## Column Renaming
def rename_age_columns(column_name):

    age_bracket = column_name.split()[1].replace("-", "_")

    new_column_name = f"N_{age_bracket}"
    return new_column_name
