import arviz as az
import pickle
import pandas as pd
from pathlib import Path
import datetime
import src.scripts.file_paths as fp
from src.scripts import utils
import numpy as np

from highcharts_core.chart import Chart
from highcharts_core.options import HighchartsOptions
from highcharts_core.options.series.area import LineSeries
from highcharts_core.options.series.boxplot import BoxPlotSeries, BoxPlotData
from highcharts_core.options.series.scatter import ScatterSeries
from highcharts_core.highcharts import SharedOptions


first_pref_or_2pp = "first_preference"

if first_pref_or_2pp == "first_preference":
    party_columns = ["ALP", "LNP", "GRN"]
    party_plot_names = dict(ALP="ALP", LNP="L/NP", GRN="Greens")
    time_filename = fp.html_filename_time_first_preference
    election_tomorrow_filename = fp.html_filename_election_tomorrow_first_preference
    title_start = "First Preference"
elif first_pref_or_2pp == "2pp":
    party_columns = ["ALP_2pp", "LNP_2pp"]
    party_plot_names = dict(ALP_2pp="ALP (2PP)", LNP_2pp="LNP (2PP)")
    time_filename = fp.html_filename_time_2pp
    election_tomorrow_filename = fp.html_filename_election_tomorrow_2pp
    title_start = "Two Party Preferred"
else:
    raise NameError("Must be one of 'first_preference' or '2pp'")


date = "20241128"

# Filenames
netcdf_filename, model_data_filename, model_df_filename, election_data_filename = (
    fp.get_filenames(date, first_pref_or_2pp)
)
print(f"Using model fit {netcdf_filename}...")

output_folder = Path(f"src/results/{date}")
output_folder.mkdir(exist_ok=True)
election_date = datetime.datetime(2022, 5, 21)

# Load the data
trace = az.from_netcdf(f"{netcdf_filename}")
with open(model_data_filename, "rb") as f:
    data = pickle.load(f)
df = pd.read_csv(model_df_filename)

# Change the x-axis to have the right format
df["x_axis_date"] = pd.to_datetime(df["PollDate"])
df["x_axis_date"] = (df.x_axis_date.astype(np.int64) / 10**6).astype(np.int64)

df["PollName"] = df["PollName"].str.replace("_", " ")

df = df.loc[df.Scope == "NAT"]
pollsters = df.PollName.unique()
parties = df.Party.unique()
series_identifiers = [(pollster, party) for pollster in pollsters for party in parties]


# Make the chart from our options file
shared_options = SharedOptions.from_js_literal(fp.highcharts_shared_options)
shared_options_snippet = shared_options.to_js_literal()


# Make a plot of the first preference percentages for each party if the election was held today
last_time = trace.posterior.coords["time"].values[-1]
support = trace.posterior.mu.sel(time=last_time)
lower_quantiles = support.quantile(0.025, ("chain", "draw"))
upper_quantiles = support.quantile(0.975, ("chain", "draw"))

lower_sigma = support.quantile(0.16, ("chain", "draw"))
upper_sigma = support.quantile(0.84, ("chain", "draw"))

medians = support.quantile(0.5, ("chain", "draw"))

# # Make the chart from our options file
chart_options = HighchartsOptions.from_js_literal(fp.election_held_today_options)

# Add the different parties
for i, p in enumerate(party_columns):

    vals = np.round(
        [
            100 * lower_quantiles.sel(party=f"{p}_NAT"),
            100 * lower_sigma.sel(party=f"{p}_NAT"),
            100 * medians.sel(party=f"{p}_NAT"),
            100 * upper_sigma.sel(party=f"{p}_NAT"),
            100 * upper_quantiles.sel(party=f"{p}_NAT"),
        ],
        1,
    )
    d = dict(zip(["low", "q1", "median", "q3", "high"], vals))
    d["x"] = i
    d["color"] = utils.get_colour(p)
    d["fill_color"] = utils.get_colour(p)
    d["name"] = p
    d = BoxPlotData.from_dict(d)
    series = BoxPlotSeries.from_dict(dict(data=d))
    #     name=p, data=data, color=utils.get_colour(p), fill_color=utils.get_colour(p)
    # )
    chart_options.add_series(series)

chart_options.title.text = (
    f"{title_start} Voting Intention if an election were held tomorrow"
)
chart = Chart.from_options(chart_options)
chart.container = "container"
js_code = chart.to_js_literal()

# Insert our fillColor strings
# Surely a better way to do this...
for p in party_columns:
    index = js_code.find(f"name: '{p}'")
    js_code = f"{js_code[:index]}fillColor: '{utils.get_colour(p)}',\n{js_code[index:]}"

html_page = utils.make_html(js_code, shared_options_snippet, height=300)
with open(election_tomorrow_filename, "w") as f:
    f.write(html_page)

## Plot of Support over time
chart_options = HighchartsOptions.from_js_literal(fp.polls_over_time_options)

df["PollResult"] = np.round(df["PollResult"], 1)
pollster_symbols = utils.get_pollster_markers(backend="HighCharts")
# Add the different polls
for s in series_identifiers:

    pollster = s[0]
    party = s[1]
    plot_name = party_plot_names[party]
    symbol_dict = pollster_symbols[pollster.replace(" ", "_")]
    mask = (df.PollName == pollster) & (df.Party == party)
    ss = ScatterSeries.from_pandas(
        df.loc[mask],
        property_map=dict(x="x_axis_date", y="PollResult", id="PollName"),
        color=utils.get_colour(s[1]),
        name=plot_name,
        opacity=0.5,
        marker=dict(symbol=symbol_dict["marker"], radius=symbol_dict["size"]),
    )
    chart_options.add_series(ss)

# Add the support levels over time
x_coords = trace.posterior.coords["time"].values
xx_dates = [
    (election_date + datetime.timedelta(days=int(x))).strftime("%Y-%m-%d")
    for x in x_coords
]
xx = pd.to_datetime(xx_dates)
xx = (xx.astype(np.int64) / 10**6).astype(np.int64)
for party in parties:

    name = party_plot_names[party]
    mean_values = (
        trace.posterior.mu.mean(("chain", "draw")).sel(party=f"{party}_NAT").values
        * 100,
    )
    y = np.round(mean_values[0], 2).tolist()
    data_values = [(xval, yval) for xval, yval in zip(xx, y)]
    series = LineSeries(
        name=name,
        data=data_values[::4],
        color=utils.get_colour(party[:3]),
        id=f"{name} Voting Intention",
        lineWidth=8,
    )
    chart_options.add_series(series)


chart_options.title.text = f"{title_start} Voting Intention over time"
chart = Chart.from_options(chart_options)
chart.container = "container"
js_code = chart.to_js_literal()

html_page = utils.make_html(js_code, shared_options_snippet)
with open(time_filename, "w") as f:
    f.write(html_page)
