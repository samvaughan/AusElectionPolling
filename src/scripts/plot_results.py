import matplotlib.pyplot as plt
import arviz as az
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
import datetime
import matplotlib.dates as mdates
from src.scripts import utils
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("yaml_file")
args = parser.parse_args()


with open(args.yaml_file, "r") as f:
    params = yaml.safe_load(f)

# Initial Data
date = "20241118"
party_long_name = params["party_long_name"]
party_column_name = params["party_column_name"]
colour = params["main_colour"]

netcdf_filename, model_data_filename, model_df_filename, election_data_filename = (
    utils.get_filenames(party_column_name, date)
)

output_folder = Path(f"src/results/{date}")
output_folder.mkdir(exist_ok=True)
election_date = datetime.datetime(2022, 5, 21)


# Load the data
trace = az.from_netcdf(netcdf_filename)
with open(model_data_filename, "rb") as f:
    data = pickle.load(f)

df = pd.read_csv(model_df_filename)

# Make the y errors
y_errors = np.sqrt(
    data["poll_result"] * (1 - data["poll_result"]) / data["survey_size"]
)

mu = trace.posterior.mu
fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

xx = [election_date + datetime.timedelta(days=x) for x in data["N_days"]]
ax.errorbar(
    xx,
    data["poll_result"] * 100,
    yerr=100 * y_errors,
    c="k",
    linewidth=1.0,
    linestyle="None",
    marker="None",
    markersize=5,
    alpha=0.3,
)
for i in range(data["N_polls"]):
    ax.scatter(
        xx[i],
        data["poll_result"].iloc[i] * 100,
        c="k",
        marker=df["MarkerShape"].iloc[i],
        s=data["survey_size"].iloc[i] / 20,
        alpha=0.3,
    )

xx = [
    election_date + datetime.timedelta(days=x) for x in range(data["prediction_date"])
]
ax.plot(xx, mu.mean(dim=("chain", "draw")) * 100, linewidth=3.0, c=colour)
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
ax.set_ylabel("Vote (%)", fontsize="large")
ax.set_title(f"{party_long_name} Support", fontsize="xx-large", loc="left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%b"))
# Rotates and right-aligns the x labels so they don't crowd each other.
for label in ax.get_xticklabels(which="major"):
    label.set(rotation=30, horizontalalignment="right")

fig.savefig(f"{output_folder}/{party_column_name}_{date}_latent_support.pdf")


# Plot the house effects
house_means = trace.posterior.house_effects.mean(("chain", "draw"))
sorted_pollsters = trace.posterior["house_effects"].sortby(house_means)
labeller = az.labels.MapLabeller(var_name_map={"house_effects": ""})
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
)
ax.set_xlabel("House Effect (percentage points)")
labels = []
for label in ax.get_yticklabels():
    labels.append(label.get_text().replace("[", "").replace("]", "").replace("_", " "))
ax.set_yticklabels(labels)
fig.savefig(f"{output_folder}/{party_column_name}_{date}_house_effects.pdf")


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
fig.savefig(f"{output_folder}/{party_column_name}_{date}_swing.pdf")
