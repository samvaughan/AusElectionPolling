import pandas as pd
import cmdstanpy
import arviz as az
import numpy as np
from datetime import datetime
import pickle
import yaml
import argparse
from src.scripts import utils

parser = argparse.ArgumentParser()
parser.add_argument("yaml_file")
args = parser.parse_args()


with open(args.yaml_file, "r") as f:
    params = yaml.safe_load(f)

date = datetime.today().strftime("%Y%m%d")
party_long_name = params["party_long_name"]
party_column_name = params["party_column_name"]
additional_variance = params["additional_variance"]

# make the filenames
netcdf_filename, model_data_filename, model_df_filename, election_data_filename = (
    utils.get_filenames_single_party(party_column_name, date)
)

print(f"Saving to {netcdf_filename}...")

df = pd.read_csv(
    "src/data/Polls/poll_data_latest.csv",
    parse_dates=["StartDate", "EndDate"],
    index_col=0,
)
# Only select national polls here
df = df.loc[df.Scope == "NAT"]

# Encode the pollster
df["PollName"] = pd.Categorical(df["PollName"])
# Note that we don't have to subtract 1 here as the 0th category is the election
df["PollName_Encoded"] = df["PollName"].cat.codes

# Take the poll as occuring at the middle of the range
# Can do a better job here...
df["PollDate"] = df.StartDate + 0.5 * (df.EndDate - df.StartDate)

election_data = df.iloc[0]
df = df.drop(0)

# Find the time since the election
df["N_Days"] = (df.EndDate - df.StartDate.iloc[0]).dt.days.astype(int)

# Coerce to integers
df["Sample"] = df["Sample"].astype(int)
df["PollError"] = np.sqrt(
    (df[party_column_name] / 100) * (1 - df[party_column_name] / 100) / df["Sample"]
)

incumbent = (party_column_name == "ALP") | (party_column_name == "ALP_2pp")

# Add a different marker style for each pollster
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
df["MarkerShape"] = df.PollName.map(marker_dict)

# Ignore the election itself
N_pollsters = len(df.PollName_Encoded.unique())

# Date we want to have our predictions on
prediction_date = (datetime.today() - election_data.StartDate).days

# Drop the NAs if they exist
df = df.dropna(subset=[party_column_name])


# Now lay out all of the data
data = dict(
    N_polls=len(df),
    N_pollsters=N_pollsters,
    N_days=df["N_Days"],
    PollName_Encoded=df["PollName_Encoded"],
    prediction_date=prediction_date,
    poll_error=df["PollError"],
    survey_size=df["Sample"],
    poll_result=(df[party_column_name] / 100),
    election_result_2022=election_data[party_column_name] / 100,
    additional_variance=additional_variance,
    incumbent=incumbent,
)

model = cmdstanpy.CmdStanModel(stan_file="src/scripts/latent_vote.stan")

fit = model.sample(data=data, max_treedepth=14, adapt_delta=0.95)


# Turn this into inference data
coords = {
    "pollster": df["PollName"].cat.categories[1:],
    "time": np.arange(data["prediction_date"]),
}
dims = {
    "mu": ["time"],
    "house_effects": ["pollster"],
}
trace = az.from_cmdstanpy(
    posterior=fit,
    coords=coords,
    dims=dims,
)

# Save the inference data
az.to_netcdf(trace, netcdf_filename)

# And the model data
with open(model_data_filename, "wb") as f:
    pickle.dump(data, f)

# And our processed poll array
df.to_csv(model_df_filename, index=False)

# And the election data
election_data.to_csv(election_data_filename)
