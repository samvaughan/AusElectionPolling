import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
import arviz as az
import pandas as pd

import src.scripts.file_paths as fp


def get_pollster_markers(backend="matplotlib"):

    if backend == "matplotlib":
        marker_dict = {
            "Essential_Research": "o",
            "Freshwater_Strategy": "s",
            "Newspoll_Pyxis": "D",
            "Newspoll_YouGov": "^",
            "RedBridge_Group": "v",
            "Resolve_Strategic": "P",
            "Roy_Morgan": "X",
            "Wolf_&_Smith": "*",
            "YouGov": "H",
        }
    elif backend == "HighCharts":
        marker_dict = {
            "Essential_Research": dict(marker="circle", size=7),
            "Freshwater_Strategy": dict(marker="square", size=5),
            "Newspoll_Pyxis": dict(marker="diamond", size=5),
            "Newspoll_YouGov": dict(marker="triangle", size=5),
            "RedBridge_Group": dict(marker="triangle-down", size=5),
            "Resolve_Strategic": dict(marker="circle", size=5),
            "Roy_Morgan": dict(marker="square", size=7),
            "Wolf_&_Smith": dict(marker="diamond", size=7),
            "YouGov": dict(marker="triangle", size=7),
        }
    else:
        raise NameError("Backend type not understood")
    return marker_dict


def get_correlation_matrix(unique_parameters):

    if (len(unique_parameters) == 2) & ("ALP_2pp_NAT" in unique_parameters):
        correlation_matrix = pd.DataFrame(
            dict(ALP_2pp=[1, -1], LNP_2pp=[-1, 1]), index=["ALP_2pp", "LNP_2pp"]
        )
    else:
        vote_shares = pd.read_csv(
            fp.three_party_voteshare_data, index_col="UniqueIdentifier"
        ).drop(["PartyAb", "StateAb", "DivisionNm"], axis=1)
        vote_shares = vote_shares.loc[unique_parameters]

        correlation_matrix = vote_shares.T.corr()

    return correlation_matrix


# Make the data frames
def make_data_for_fitting(df, party_columns, states, prediction_date):

    unique_parameters = [f"{p}_{s}" for p in party_columns for s in states]
    election_date = df.StartDate.iloc[0]

    # Encode the parties
    N_parties = len(unique_parameters)

    # Now go from wide to long and make the 'part index' column
    df = df.melt(
        value_vars=party_columns,
        id_vars=[
            "StartDate",
            "EndDate",
            "PollName",
            "Mode",
            "Scope",
            "Sample",
        ],
        var_name="Party",
        value_name="PollResult",
    )

    # Drop the states and parties we don't care about
    df = df.loc[df.Party.isin(party_columns)]
    df = df.loc[df.Scope.isin(states)]

    # Have a unique index for each party in each state
    df["Party_Scope"] = df["Party"] + "_" + df["Scope"]
    df["Party_Scope"] = pd.Categorical(df.Party_Scope, categories=unique_parameters)
    df["PartyIndex"] = df["Party_Scope"].cat.codes + 1

    # Strip off the election values
    election_data = df.loc[df.PollName == "Election"].sort_values("PartyIndex")
    df = df.drop(election_data.index)

    ## Time based manipulations
    # Take the poll as occuring at the end of the range
    # Can do a better job here...
    df["PollDate"] = df.EndDate

    # Date we want to have our predictions on
    prediction_date_integer = (prediction_date - election_date).days

    # Add a different marker style for each pollster
    df["MarkerShape"] = df.PollName.map(get_pollster_markers())

    # Find the time since the election for each poll
    df["N_Days"] = (df.PollDate - election_date).dt.days.astype(int)

    # Encode the pollster
    df["PollName"] = pd.Categorical(df["PollName"])
    df["PollName_Encoded"] = df["PollName"].cat.codes + 1
    N_pollsters = len(df.PollName_Encoded.unique())

    # Make the errors for each poll
    # Coerce to integers
    df["Sample"] = df["Sample"].astype(int)
    # Make the variance
    r = df["PollResult"].values / 100
    df["PollVariance"] = r * (1 - r) / df["Sample"].values

    # And the Cholesky decomposition of the covariance matrix
    correlation_matrix = get_correlation_matrix(unique_parameters).values

    # Add a tiny number to the diagonals to deal with machine precision issues
    eps = 1e-12
    positive_offset = np.eye(N_parties) * eps
    correlation_matrix = correlation_matrix + positive_offset
    cholesky_matrix_loc = np.linalg.cholesky(correlation_matrix)
    cholesky_matrix_uncertainty = np.full((N_parties, N_parties), 0.01)

    # cholesky_matrix = np.linalg.cholesky(correlation_matrix)
    df = df.sort_values("N_Days")

    # And finally drop polls with NaN values
    df = df.dropna(subset="PollResult")

    # Now lay out all of the data
    data = dict(
        N_polls=len(df),
        N_pollsters=N_pollsters,
        N_parties=N_parties,
        N_days=df["N_Days"],
        party_index=df["PartyIndex"],
        PollName_Encoded=df["PollName_Encoded"],
        prediction_date=prediction_date_integer,
        poll_variance=df["PollVariance"],
        survey_size=df["Sample"],
        poll_result=df["PollResult"] / 100,
        election_result=election_data.PollResult.values / 100,
        election_result_index=election_data.PartyIndex.values,
        inflator=np.sqrt(2),
        correlation_matrix=correlation_matrix,
        cholesky_matrix_loc=cholesky_matrix_loc,
        cholesky_matrix_scale=cholesky_matrix_uncertainty,
    )

    return data, df, election_data, unique_parameters


# HTML


def make_html(js_code, js_settings, height=600):

    html_page = f"""
    <!DOCTYPE html>

    <html>

    <head>
        <link type="text/css" rel="stylesheet" href="TASK3.css" media="all" />
        <script language="javascript" type="text/javascript" src="https://code.highcharts.com/highcharts.js"></script>
        <script src="https://code.highcharts.com/highcharts-more.js"></script>
        <script language="javascript" type="text/javascript" src="https://code.highcharts.com/modules/exporting.js"></script>
    </head>

    <body>

        <div id="container" style="min-width: 310px; max-width: 1200px; height: {height}px; margin: 0 auto">

        <script type="text/javascript">

        {js_settings}
        Highcharts.SVGRenderer.prototype.symbols.cross = function (x, y, w, h) {{
                return ['M', x, y, 'L', x + w, y + h, 'M', x + w, y, 'L', x, y + h, 'z'];
            }};
            if (Highcharts.VMLRenderer) {{
                Highcharts.VMLRenderer.prototype.symbols.cross = Highcharts.SVGRenderer.prototype.symbols.cross;
            }}
        Highcharts.SVGRenderer.prototype.symbols.plus = function (x, y, w, h) {{
                return ['M', x, y + h/2, 'L', x + w, y+h/2, 'M', x + w/2, y, 'L', x+w/2, y + h, 'z'];
            }};
            if (Highcharts.VMLRenderer) {{
                Highcharts.VMLRenderer.prototype.symbols.cross = Highcharts.SVGRenderer.prototype.symbols.cross;
            }}

        {js_code}

        </script>
        </div>
    </body>

    </html>
    """

    return html_page


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
    return colours[party_name[:3]]


def make_all_state_space_plots(trace, data, df, election_date, parties, title):

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for p, party in enumerate(parties):
        colour = get_colour(party)
        tmp_data = data.copy()
        party_mask = (data["party_index"] == p + 1).reset_index(drop=True).values
        tmp_data["poll_result"] = data["poll_result"].loc[party_mask]
        tmp_data["N_days"] = data["N_days"].loc[party_mask]
        tmp_data["poll_variance"] = data["poll_variance"].loc[party_mask]
        tmp_data["survey_size"] = data["survey_size"].loc[party_mask]
        tmp_data["poll_variance"] = data["poll_variance"].loc[party_mask]

        make_state_space_plot(
            trace.sel(party=party),
            tmp_data,
            df.loc[party_mask],
            election_date,
            colour=colour,
            y_errors=np.sqrt(tmp_data["poll_variance"]),
            fig=fig,
            ax=ax,
            label=party,
        )
    ax.set_ylabel("Vote (%)", fontsize="large")
    ax.set_title(title, fontsize="xx-large", loc="left")
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
    for i in range(len(data["poll_result"])):
        ax.scatter(
            xx[i],
            data["poll_result"].iloc[i] * 100,
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


# make a plot for all states in a given party
def plot_state_space_debug(
    trace,
    df,
    election_data,
    party_name,
    states=["NAT", "VIC", "NSW", "WA", "SA", "QLD"],
    ci_interval=0.95,
):
    alpha = (1 - ci_interval) / 2
    mu = trace.posterior.mu
    means = mu.mean(("chain", "draw")) * 100
    lower = mu.quantile(alpha, dim=("chain", "draw")) * 100
    upper = mu.quantile(1 - alpha, dim=("chain", "draw")) * 100

    fig, axs = plt.subplots(constrained_layout=True, ncols=3, nrows=2, figsize=(17, 5))

    x_coords = trace.posterior.coords["time"].values
    for ax, state in zip(axs.ravel(), states):
        name = f"{party_name}_{state}"
        polls = df.loc[df.Party_Scope == name]
        ax.plot(
            x_coords,
            means.sel(party=name),
            linewidth=3.0,
            label=state,
            zorder=10,
            c=get_colour(name),
        )
        ax.fill_between(
            x_coords,
            lower.sel(party=name),
            upper.sel(party=name),
            alpha=0.3,
            facecolor=get_colour(name),
        )
        ax.set_title(state)
        ax.errorbar(
            polls["N_Days"],
            polls["PollResult"],
            yerr=np.sqrt(polls.PollVariance) * 100,
            marker="None",
            linestyle="None",
            zorder=1,
            c="k",
        )
        ax.scatter(polls["N_Days"], polls["PollResult"], zorder=2, c="k")

        # add the election result
        ax.scatter(
            [0],
            [election_data.loc[election_data.Party_Scope == name, "PollResult"]],
            s=150,
            c="orange",
            zorder=10,
        )

    # labelLines(ax.get_lines(), zorder=2.5)
    fig.suptitle(party_name)

    return fig, axs
