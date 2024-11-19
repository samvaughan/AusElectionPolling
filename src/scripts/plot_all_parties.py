import arviz as az
import pickle
import pandas as pd
from pathlib import Path
import datetime
from src.scripts import utils


first_pref_or_2pp = "first_preference"
if first_pref_or_2pp == "first_preference":
    party_columns = ["ALP", "LNP", "GRN", "PHON"]
elif first_pref_or_2pp == "2pp":
    party_columns = ["ALP_2pp", "LNP_2pp"]
else:
    raise NameError("Must be one of 'first_preference' or '2pp'")


date = "20241119"

# Filenames
netcdf_filename, model_data_filename, model_df_filename, election_data_filename = (
    utils.get_filenames(date, first_pref_or_2pp)
)
print(f"Using model fit {netcdf_filename}...")

output_folder = Path(f"src/results/{date}")
output_folder.mkdir(exist_ok=True)
election_date = datetime.datetime(2022, 5, 21)


# Load the data
trace = az.from_netcdf(netcdf_filename)
with open(model_data_filename, "rb") as f:
    data = pickle.load(f)
df = pd.read_csv(model_df_filename)

utils.make_all_state_space_plots(trace, data, df, election_date, party_columns)