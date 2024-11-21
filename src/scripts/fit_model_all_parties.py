import pandas as pd
import cmdstanpy
import arviz as az
import numpy as np
from datetime import datetime
import pickle

# import argparse
from src.scripts import utils

# parser = argparse.ArgumentParser()
# parser.add_argument("yaml_file")
# args = parser.parse_args()


# with open(args.yaml_file, "r") as f:
#     params = yaml.safe_load(f)

first_pref_or_2pp = "first_preference"

date = datetime.today().strftime("%Y%m%d")
# party_long_name = params["party_long_name"]
# party_column_name = params["party_column_name"]
# additional_variance = params["additional_variance"]

# make the filenames
netcdf_filename, model_data_filename, model_df_filename, election_data_filename = (
    utils.get_filenames(date, first_pref_or_2pp=first_pref_or_2pp)
)
print(f"Saving fit to {netcdf_filename}\n")


df = pd.read_csv(
    "src/data/Polls/poll_data_latest.csv",
    parse_dates=["StartDate", "EndDate"],
    index_col=0,
)
# Only select national polls here
df = df.loc[df.Scope == "NAT"]


if first_pref_or_2pp == "first_preference":
    party_columns = ["ALP", "LNP", "GRN", "PHON"]
    incumbent = [1, 0, 0, 0]
elif first_pref_or_2pp == "2pp":
    party_columns = ["ALP_2pp", "LNP_2pp"]
    incumbent = [1, 0]
else:
    raise NameError("Must be one of 'first_preference' or '2pp'")

# Encode the pollster
df["PollName"] = pd.Categorical(df["PollName"])
# Note that we don't have to subtract 1 here as the 0th category is the election
df["PollName_Encoded"] = df["PollName"].cat.codes

# Encode the parties
N_parties = len(party_columns)

# Take the poll as occuring at the middle of the range
# Can do a better job here...
df["PollDate"] = df.EndDate

election_data = df.loc[df.PollName == "Election"]
df = df.drop(election_data.index)

# Find the time since the election
df["N_Days"] = (df.EndDate - df.StartDate.iloc[0]).dt.days.astype(int)


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
prediction_date = (datetime.today() - election_data.loc[0, "StartDate"]).days

# Drop the NAs if they exist
df = df.dropna(subset=party_columns)

# Make the errors for each poll
# Coerce to integers
df["Sample"] = df["Sample"].astype(int)
# Make the variance
r = df.loc[:, party_columns].values / 100
poll_variance = r * (1 - r) / df["Sample"].values[:, None]

# Now lay out all of the data
data = dict(
    N_polls=len(df),
    N_pollsters=N_pollsters,
    N_parties=N_parties,
    N_days=df["N_Days"],
    PollName_Encoded=df["PollName_Encoded"],
    prediction_date=prediction_date,
    poll_variance=poll_variance,
    survey_size=df["Sample"],
    poll_result=(df.loc[:, party_columns].values / 100),
    election_result_2022=election_data.loc[:, party_columns].values.squeeze() / 100,
    additional_variance=0.001,
    incumbent=incumbent,
    inflator=np.sqrt(2),
)

model = cmdstanpy.CmdStanModel(
    stan_file="src/scripts/stan/latent_vote_all_parties.stan"
)

fit = model.sample(data=data)


# Turn this into inference data
coords = {
    "pollster": df["PollName"].cat.categories[1:],
    "time": np.arange(data["prediction_date"]),
    "party": party_columns,
}
dims = {
    "mu": ["time", "party"],
    "house_effects": ["pollster", "party"],
    "sigma": ["party"],
    "time_variation_sigma": ["time"],
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
