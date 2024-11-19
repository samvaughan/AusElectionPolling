import pandas as pd
import cmdstanpy
import arviz as az
import numpy as np
from datetime import datetime
from src.scripts import utils

start_date = datetime.strptime("2004-10-09", "%Y-%m-%d")
end_date = datetime.strptime("2007-11-24", "%Y-%m-%d")

election_start = 37.64
election_end = 43.38

df = pd.read_csv(
    "src/data/Polls/AustralianElectionPolling_2004_2007.csv",
    parse_dates=["startDate", "endDate"],
    index_col=0,
)

# Encode the pollster
df["PollName"] = pd.Categorical(df["org"])
# Note that we don't have to subtract 1 here as the 0th category is the election
df["PollName_Encoded"] = df["PollName"].cat.codes

# Take the poll as occuring at the middle of the range
# Can do a better job here...
df["PollDate"] = df.endDate

# election_data = df.iloc[0]
# df = df.drop(0)

# Find the time since the election
df["N_Days"] = (df.endDate - start_date).dt.days.astype(int)

# Coerce to integers
df["Sample"] = df["sampleSize"].astype(int)

# Add a different marker style for each pollster
marker_dict = {
    "Morgan, F2F": "o",
    "Newspoll": "s",
    "Nielsen": "D",
    "Galaxy": "X",
    "Morgan, Phone": "*",
}
df["MarkerShape"] = df.PollName.map(marker_dict)

# Ignore the election itself
N_pollsters = len(df.PollName_Encoded.unique())

# Date we want to have our predictions on
prediction_date = (end_date - start_date).days

# Drop the NAs if they exist
df = df.dropna(subset=["ALP"])


# Now lay out all of the data
data = dict(
    N_polls=len(df),
    N_pollsters=N_pollsters,
    N_days=df["N_Days"],
    PollName_Encoded=df["PollName_Encoded"] + 1,
    prediction_date=prediction_date,
    survey_size=df["Sample"],
    poll_result=(df["ALP"] / 100),
    election_result_start=election_start / 100,
    election_result_finish=election_end / 100,
    additional_variance=0.0,
)

model = cmdstanpy.CmdStanModel(
    stan_file="src/scripts/latent_vote_between_elections.stan"
)

fit = model.sample(data=data, max_treedepth=14)


# Turn this into inference data
coords = {
    "pollster": df["PollName"].cat.categories,
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

# Make a few plots
fig, ax = utils.make_house_effects_plot(trace, colour="red")
fig.savefig("src/results/2004_2007/house_effects.png")
# Now make the plots
fig, ax = utils.make_state_space_plot(
    trace, data, df, start_date, colour="red", party_long_name="ALP 2pp"
)
fig.savefig("src/results/2004_2007/state_space.png")

# # Save the inference data
# az.to_netcdf(trace, netcdf_filename)

# # And the model data
# with open(model_data_filename, "wb") as f:
#     pickle.dump(data, f)

# # And our processed poll array
# df.to_csv(model_df_filename, index=False)

# # And the election data
# election_data.to_csv(election_data_filename)
