import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
import arviz as az
from pathlib import Path


def get_filenames(date, first_pref_or_2pp):

    assert first_pref_or_2pp in [
        "first_preference",
        "2pp",
    ], "Must be one of first_preference or 2pp"

    # Main folder for inference data
    master_folder_inference = Path(f"src/data/InferenceData/{date}/")
    master_folder_inference.mkdir(exist_ok=True)

    # Main folder for model data
    master_folder_Model = Path(f"src/data/ModelData/{date}/")
    master_folder_Model.mkdir(exist_ok=True)

    netcdf_filename = master_folder_inference / f"{first_pref_or_2pp}_{date}"
    model_data_filename = master_folder_Model / f"{first_pref_or_2pp}_{date}.pkl"
    model_df_filename = master_folder_Model / f"{first_pref_or_2pp}_{date}.csv"
    election_data_filename = master_folder_Model / "election_data.csv"

    return (
        netcdf_filename,
        model_data_filename,
        model_df_filename,
        election_data_filename,
    )


def get_filenames_single_party(party_short_name, date):

    # Main folder for inference data
    master_folder_inference = Path(f"src/data/InferenceData/{date}/")
    master_folder_inference.mkdir(exist_ok=True)

    # Main folder for model data
    master_folder_Model = Path(f"src/data/ModelData/{date}/")
    master_folder_Model.mkdir(exist_ok=True)

    netcdf_filename = master_folder_inference / f"{party_short_name}_{date}"
    model_data_filename = master_folder_Model / f"{party_short_name}_{date}.pkl"
    model_df_filename = master_folder_Model / f"{party_short_name}_{date}.csv"
    election_data_filename = master_folder_Model / "election_data.csv"

    return (
        netcdf_filename,
        model_data_filename,
        model_df_filename,
        election_data_filename,
    )


# Plotting


def get_colour(party_name):

    colours = dict(
        ALP="#e53440",
        ALP_2pp="#e53440",
        LNP="#1c4f9c",
        LNP_2pp="#1c4f9c",
        GRN="#008943",
        PHON="#f36c21",
    )
    return colours[party_name]


def make_all_state_space_plots(trace, data, df, election_date, parties):

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for p, party in enumerate(parties):
        colour = get_colour(party)
        tmp_data = data.copy()
        tmp_data["poll_result"] = data["poll_result"][:, p]

        make_state_space_plot(
            trace.sel(party=party),
            tmp_data,
            df,
            election_date,
            colour=colour,
            y_errors=np.sqrt(data["poll_variance"])[:, p],
            fig=fig,
            ax=ax,
            label=party,
        )
    ax.set_ylabel("Vote (%)", fontsize="large")
    ax.set_title("First Preference Voting Intention", fontsize="xx-large", loc="left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%b"))
    ax.legend()
    # Rotates and right-aligns the x labels so they don't crowd each other.
    for label in ax.get_xticklabels(which="major"):
        label.set(rotation=30, horizontalalignment="right")
    return fig, ax


def make_state_space_plot(
    trace,
    data,
    df,
    election_date,
    colour,
    y_errors,
    fig=None,
    ax=None,
    label=None,
):
    mu = trace.posterior.mu

    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

    xx = [election_date + datetime.timedelta(days=x) for x in data["N_days"]]
    ax.errorbar(
        xx,
        data["poll_result"] * 100,
        yerr=100 * y_errors,
        c=colour,
        linewidth=1.0,
        linestyle="None",
        marker="None",
        markersize=5,
        alpha=0.3,
    )
    for i in range(data["N_polls"]):
        ax.scatter(
            xx[i],
            data["poll_result"][i] * 100,
            c=colour,
            marker=df["MarkerShape"].iloc[i],
            s=data["survey_size"].iloc[i] / 20,
            alpha=0.3,
        )

    xx = [
        election_date + datetime.timedelta(days=x)
        for x in range(data["prediction_date"])
    ]
    ax.plot(
        xx, mu.mean(dim=("chain", "draw")) * 100, linewidth=3.0, c=colour, label=label
    )
    ax.fill_between(
        xx,
        mu.quantile(0.025, dim=("chain", "draw")) * 100,
        mu.quantile(0.975, dim=("chain", "draw")) * 100,
        alpha=0.1,
        facecolor=colour,
    )
    ax.fill_between(
        xx,
        mu.quantile(0.16, dim=("chain", "draw")) * 100,
        mu.quantile(0.84, dim=("chain", "draw")) * 100,
        alpha=0.3,
        facecolor=colour,
    )
    return fig, ax


def make_all_house_effects_plot(trace, parties):

    if len(parties) == 2:
        fig, axs = plt.subplots(layout="constrained", ncols=2)
    else:
        fig, axs = plt.subplots(layout="constrained", figsize=(15, 7), ncols=2, nrows=2)

    for ax, party in zip(axs.ravel(), parties):
        colour = get_colour(party)
        make_house_effects_plot(trace.sel(party=party), colour=colour, fig=fig, ax=ax)
        ax.set_title("")
    return fig, ax


def make_house_effects_plot(trace, colour, fig=None, ax=None):
    # Plot the house effects
    house_means = trace.posterior.house_effects.mean(("chain", "draw"))
    sorted_pollsters = trace.posterior["house_effects"].sortby(
        house_means, ascending=False
    )
    labeller = az.labels.MapLabeller(var_name_map={"house_effects": ""})
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(layout="constrained")

    az.plot_forest(
        sorted_pollsters,
        var_names=["house_effects"],
        combined=True,
        transform=lambda x: x * 100,
        ax=ax,
        labeller=labeller,
        colors=[colour],
        linewidth=3.0,
        markersize=10,
        hdi_prob=None,
    )
    ax.set_xlabel("House Effect (percentage points)")
    labels = []
    for label in ax.get_yticklabels():
        labels.append(
            label.get_text().replace("[", "").replace("]", "").replace("_", " ")
        )
    ax.set_yticklabels(labels)

    return fig, ax


def make_swing_plot(trace, data, colour, party_long_name, fig=None, ax=None):

    # Plot the swing since the last election
    swing = 100 * (
        trace.posterior.mu.sel(time=data["prediction_date"] - 1).stack(
            sim=("chain", "draw")
        )
        - data["election_result_2022"]
    )
    hdi = az.hdi(
        100
        * (
            trace.posterior.mu.sel(time=data["prediction_date"] - 1)
            - data["election_result_2022"]
        ),
        hdi_prob=0.9,
    )

    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(layout="constrained")

    az.plot_dist(
        swing,
        kind="kde",
        rug=True,
        fill_kwargs=dict(alpha=0.3, color=colour),
        color=colour,
        rug_kwargs=dict(alpha=0.1, space=-1.1),
        ax=ax,
    )
    ax.lines[1].set_linewidth(3.0)
    ax.set_yticks([])
    ax.set_xlabel("Swing since 2022 election (%)", fontsize="large")
    ax.axvline(0.0, c="k", linestyle="dashed", linewidth=2.0)
    ax.set_title(
        f"{party_long_name} swing\nMean = {swing.mean():.2f}%\n90% CI = ({hdi.mu.values[0]:.2f}%,{hdi.mu.values[1]:.2f}%)",
        fontsize="xx-large",
        loc="left",
    )

    return fig, ax


def make_current_latent_value_plot(trace, parties):

    colours = [get_colour(p) for p in trace.posterior.coords["party"].values]
    latest_date = trace.posterior.coords["time"].values.max()
    if len(parties) == 2:
        fig, axs = plt.subplots(layout="constrained", ncols=2)
    else:
        fig, axs = plt.subplots(layout="constrained", figsize=(15, 7), ncols=2, nrows=2)
    for ax, party, colour in zip(axs.ravel(), parties, colours):

        az.plot_dist(
            100 * trace.sel(party=party, time=latest_date).posterior.mu,
            ax=ax,
            kind="kde",
            rug=True,
            fill_kwargs=dict(alpha=0.3, color=colour),
            color=colour,
            rug_kwargs=dict(alpha=0.1, space=-1.1),
            label=party,
        )

        ax.lines[1].set_linewidth(3.0)
        ax.set_yticks([])

    fig.suptitle("Current First Preference Voting Intention")

    return fig, ax
