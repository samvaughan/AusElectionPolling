import pandas as pd
import cmdstanpy
import arviz as az
import numpy as np
from datetime import datetime
import pickle

# import argparse
from src.scripts import utils
from src.scripts import file_paths as fp

first_pref_or_2pp = "first_preference"

prediction_date = datetime.today()
date = prediction_date.strftime("%Y%m%d")


# make the filenames
netcdf_filename, model_data_filename, model_df_filename, election_data_filename = (
    fp.get_filenames(date, first_pref_or_2pp=first_pref_or_2pp)
)
print(f"Saving fit to {netcdf_filename}\n")


df = pd.read_csv(
    fp.latest_poll_data,
    parse_dates=["StartDate", "EndDate"],
    index_col=0,
)

if first_pref_or_2pp == "first_preference":
    # party_columns = ["ALP", "LNP", "GRN"]  # PHON
    party_columns = ["ALP", "LNP", "GRN"]
    states = ["NAT", "VIC"]

elif first_pref_or_2pp == "2pp":
    party_columns = ["ALP_2pp", "LNP_2pp"]
    states = ["NAT"]

else:
    raise NameError("Must be one of 'first_preference' or '2pp'")

data, df, election_data, unique_party_state_indentifiers = utils.make_data_for_fitting(
    df, party_columns, states, prediction_date
)


model = cmdstanpy.CmdStanModel(stan_file="src/scripts/stan/MultiNormal.stan")

fit = model.sample(data=data)

# Turn this into inference data
coords = {
    "pollster": df["PollName"].cat.categories,
    "time": np.arange(data["prediction_date"]),
    "party": unique_party_state_indentifiers,
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
