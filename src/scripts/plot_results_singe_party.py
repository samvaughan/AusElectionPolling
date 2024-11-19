import arviz as az
import pickle
import pandas as pd
from pathlib import Path
import datetime
from src.scripts import utils
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("yaml_file")
args = parser.parse_args()


with open(args.yaml_file, "r") as f:
    params = yaml.safe_load(f)

# Initial Data
date = "20241119"
party_long_name = params["party_long_name"]
party_column_name = params["party_column_name"]
colour = params["main_colour"]

# Filenames
netcdf_filename, model_data_filename, model_df_filename, election_data_filename = (
    utils.get_filenames(party_column_name, date)
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


# Now make the plots
fig, ax = utils.make_state_space_plot(
    trace, data, df, election_date, colour, party_long_name
)
fig.savefig(f"{output_folder}/{party_column_name}_{date}_latent_support.png")


fig, ax = utils.make_house_effects_plot(trace, colour)
fig.savefig(f"{output_folder}/{party_column_name}_{date}_house_effects.png")

fig, ax = utils.make_swing_plot(trace, data, colour, party_long_name)
fig.savefig(f"{output_folder}/{party_column_name}_{date}_swing.png")
