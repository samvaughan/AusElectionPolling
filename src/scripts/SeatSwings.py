import numpy as np
import pandas as pd
from src.scripts import seat_utils
import matplotlib.pyplot as plt
import arviz as az
import xarray as xr

rng = np.random.default_rng(1234)

idx = pd.IndexSlice
master = pd.DataFrame()

# Now look at one division
Seat_name = "Chisholm"

for year in [2022, 2019, 2016, 2013, 2010, 2007, 2004]:

    df = pd.read_csv(
        f"src/data/Swings/HouseStateFirstPrefsByPollingPlace_{year}.csv",
        skiprows=1,
        index_col="PollingPlaceID",
    )

    totals = df.groupby("PollingPlaceID")["OrdinaryVotes"].sum()
    totals.name = "TotalOrdinaryVotes"
    df = df.join(totals)
    df["BoothPercentage"] = df["OrdinaryVotes"] / df["TotalOrdinaryVotes"] * 100
    df["VoteSigma"] = (
        np.sqrt(
            df["BoothPercentage"]
            / 100
            * (1 - df["BoothPercentage"] / 100)
            / df["OrdinaryVotes"]
        )
        * 100
    )
    df["year"] = year
    df = df.reset_index()

    # Get the difference from the national vote
    national = pd.read_csv(f"src/data/Correlations/National/{year}.csv", skiprows=1)

    # lnp_percentage = (
    #     national.loc[national.PartyAb == "LP", "TotalPercentage"]
    #     + national.loc[national.PartyAb == "NP", "TotalPercentage"]
    # )
    # row = pd.DataFrame(PartyAb="LNP", PartyNm="Lib/Nat", TotalPercentage=lnp_percentage)
    # df = df.join(national, on="PartyAb")
    # df["NationalVoteFraction"]
    df = pd.merge(
        df,
        national.loc[:, ["PartyAb", "TotalPercentage", "TotalSwing"]],
        left_on="PartyAb",
        right_on="PartyAb",
    )
    df = df.rename(
        dict(TotalPercentage="NationalPercentage", TotalSwing="NationalSwing"), axis=1
    )

    state = pd.read_csv(f"src/data/Correlations/ByState/{year}.csv", skiprows=1)
    state = state.loc[state.StateAb == "VIC"]
    df = pd.merge(
        df,
        state.loc[:, ["PartyAb", "TotalPercentage", "TotalSwing"]],
        left_on="PartyAb",
        right_on="PartyAb",
    )
    df = df.rename(
        dict(TotalPercentage="StatePercentage", TotalSwing="StateSwing"), axis=1
    )

    df["DeltafromNational"] = df["BoothPercentage"] - df["NationalPercentage"]
    df["DeltafromState"] = df["BoothPercentage"] - df["StatePercentage"]
    master = pd.concat((master, df))

master = master.set_index(["PartyAb", "DivisionNm", "PollingPlaceID", "year"])

division = master.loc[["ALP", "LP", "GRN"], :, :, :]

booth_swing = (
    division.loc[["ALP", "LP", "GRN"], "Swing"].unstack("year").dropna(subset=2022)
)

# Find the swing correlation
national_swings = (
    division.loc[["ALP", "LP", "GRN"], "NationalSwing"]
    .unstack("year")
    .dropna()
    .droplevel([1, 2])
    .drop_duplicates()
).reset_index()
state_swings = (
    division.loc[["ALP", "LP", "GRN"], "StateSwing"]
    .unstack("year")
    .dropna()
    .droplevel([1, 2])
    .drop_duplicates()
).reset_index()

national_swings["DivisionNm"] = "National"
national_swings["PollingPlaceID"] = 99999
national_swings = national_swings.set_index(["PartyAb", "DivisionNm", "PollingPlaceID"])

state_swings["DivisionNm"] = "VIC"
state_swings["PollingPlaceID"] = 99999
state_swings = state_swings.set_index(["PartyAb", "DivisionNm", "PollingPlaceID"])

swings = pd.concat((booth_swing, national_swings, state_swings))

one_division = swings.loc[idx[:, [Seat_name], :]]

# Drop NaNs- this means dropping booths which don't have data back to 2004. Will need to rethink this...
one_division = one_division.dropna().sort_index(level=["PollingPlaceID", "PartyAb"])
booth_index = one_division.index

division_plus_nat_state = pd.concat(
    (one_division, national_swings, state_swings)
).sort_index(level=["PollingPlaceID", "PartyAb"])
polling_places = (
    one_division.index.get_level_values(2).drop_duplicates().sort_values()[:-1]
)


sigma = division_plus_nat_state.std(1)
sigma = sigma.fillna(sigma.mean())

means = np.zeros_like(sigma)  # one_division.mean(1).values

corr = division_plus_nat_state.T.corr()
cov = (corr * np.outer(sigma, sigma)).values + np.eye(len(corr)) * 1e-1


# Now we have 'observed' the National and state level swings
# This comes from the model, and goes [ALP, ALP_VIC, GRN, GRN_VIC, LNP, LNP_VIC]
N = len(means)
a = np.array([-2, -1, 1, 1.5, 2, 1.5])
q = N - len(a)

mu1 = means[:q]
mu2 = means[q:]

S11 = cov[:q, :q]
S21 = cov[q:, :q]
S12 = cov[:q, q:]
S22 = cov[q:, q:]

S22_inv = np.linalg.inv(S22)
S12_S22_inv = S12 @ S22_inv

mubar = mu1 + S12_S22_inv @ (a - mu2)

N_samples = 4000

Sbar = S11 - S12_S22_inv @ S21

sim_swings = rng.multivariate_normal(mubar, Sbar, size=N_samples)
sim_swings = sim_swings.reshape(N_samples, -1, 3)

weights = (
    division.loc[idx[:, :, :, 2022]]
    .loc[booth_index, "TotalOrdinaryVotes"]
    .groupby(level=["DivisionNm", "PollingPlaceID"])
    .first()
).values

weights = weights / weights.sum()

previous_booth_percentages = (
    division.loc[idx[:, :, :, 2022]]
    .loc[booth_index, "BoothPercentage"]
    .values.reshape(-1, 3)
)

simulated_booth_percentages = previous_booth_percentages + sim_swings

simulated_seat_percentages = weights @ simulated_booth_percentages

fig, ax = plt.subplots()
ax.set_title(f"{Seat_name}")
ax.hist(simulated_seat_percentages[:, 0], color="crimson", bins="fd")
ax.hist(simulated_seat_percentages[:, 2], color="navy", bins="fd")
ax.hist(simulated_seat_percentages[:, 1], color="seagreen", bins="fd")

locs = simulated_seat_percentages.mean(0)
swings_from_2022 = weights @ mubar.reshape(-1, 3)
print("Voting Intention Swings:\n")
print(f"\tALP (national): {a[0]}%")
print(f"\tALP (Vic): {a[1]}%")
print(f"\tGRN (national): {a[2]}%")
print(f"\tGRN (Vic): {a[3]}%")
print(f"\tLNP (national): {a[4]}%")
print(f"\tLNP (Vic): {a[5]}%")
print(f"\nSeat of {Seat_name}")
print(
    f"\tSeat first preferences (mean): ALP: {locs[0]:.2f}%, GRN: {locs[1]:.2f}%, LNP: {locs[2]:.2f}%"
)
print(
    f"\tSwings from 2022: {swings_from_2022[0]:.2f}%, {swings_from_2022[1]:.2f}%, {swings_from_2022[2]:.2f}%"
)

# # This is all work with deltas

# # Find the correlation matrix between swings for the ALP, LP and GRNS in Wills
# # Note- old boundaries!!
# # division = master.loc[master.DivisionNm == "Melbourne"]

# booth_deltas = (
#     master.loc[["ALP", "LP", "GRN"], "DeltafromNational"]
#     .unstack("year")
#     .dropna(subset=2022)
# )
# PollingPlaceIDs = booth_deltas.index.levels[2]

# # Find the last swing
# means = xr.DataArray(
#     booth_deltas[2022].values, coords=dict(PollingPlace=PollingPlaceIDs)
# )

# # Find the standard deviation
# stds = booth_deltas.std(axis=1)
# # Need to impute the NaNs
# stds = stds.groupby(level=["PartyAb"]).transform(lambda x: x.fillna(x.mean()))


# # Find the correlation matrix for division
# correlation = booth_deltas.T.corr().values
# # Adjust the correlation matrix to get rid of things which only have very few election results
# ## TODO: do a better job of this
# eps = 1e-10
# correlation[~np.isfinite(correlation)] = 0.0
# correlation[correlation >= (1 - eps)] = 0
# correlation[correlation <= (-1 + eps)] = 0
# correlation += np.eye(correlation.shape[0])

# N_samples = 4000

# # Make some random swings!
# cov = (np.diag(stds) @ correlation) @ np.diag(stds)
# # Add a small value for numerical stability
# cov += np.eye(cov.shape[0]) * eps
# swings = rng.multivariate_normal(means, cov, size=N_samples)
# # Now reshape this to be 3xn
# swings = swings.reshape(N_samples, 3, -1)


# # Now work out our election result
# trace = az.from_netcdf("src/data/InferenceData/20241202/first_preference_20241202")
# mu = az.extract(trace.posterior.mu, combined=True)
# national_fracs = (
#     mu.sel(time=925, party=["ALP_NAT", "LNP_NAT", "GRN_NAT"]).to_dataarray().squeeze()
#     * 100
# )
# n = national_fracs.values.T

# # Find the weight for each booth
# weights = (
#     master.loc[idx[:, :, :, 2022], "TotalOrdinaryVotes"]
#     .groupby(level=["DivisionNm", "PollingPlaceID"])
#     .first()
# )
# weights = (weights / np.sum(weights)).values

# booths = n[..., None] + swings

# seat_percentage = booths @ weights

# fig, ax = plt.subplots()
# ax.hist(seat_percentage[:, 0], color="crimson", bins="fd")
# ax.hist(seat_percentage[:, 1], color="navy", bins="fd")
# ax.hist(seat_percentage[:, 2], color="seagreen", bins="fd")


# # plot stuff
# def plot_swings_for_seat(master, seat_name, alpha=0.5):

#     # Note- old boundaries!!
#     division = master.loc[master.DivisionNm == seat_name]

#     division_swings = (
#         division.loc[["ALP", "LP", "GRN"], "DeltafromNational"]
#         .unstack("year")
#         .dropna(subset=2022)
#     )

#     fig, ax = plt.subplots()
#     ax.plot(division_swings.loc[idx["ALP", :]].T, alpha=alpha, c="r")
#     ax.plot(division_swings.loc[idx["LP", :]].T, alpha=alpha, c="b")
#     ax.plot(division_swings.loc[idx["GRN", :]].T, alpha=alpha, c="g")
#     ax.set_xlabel("Election Date")
#     ax.set_ylabel("Booth Swing")

#     return fig, ax
