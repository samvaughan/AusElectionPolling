import pandas as pd
from pathlib import Path
import src.scripts.file_paths as fp

idx = pd.IndexSlice


def make_year_data(f):

    year = int(f.strip(".csv"))
    print(year)

    tmp = pd.read_csv(fp.correlations_DivisionFolder / f, skiprows=1)
    party_votes = tmp.groupby(["PartyAb", "StateAb", "DivisionNm"])["TotalVotes"].sum()
    total_votes = tmp.groupby(["StateAb", "DivisionNm"])["TotalVotes"].sum()

    first_pref_percentage = 100 * (party_votes / total_votes)

    # Add in the state results
    state_tmp = pd.read_csv(
        fp.correlations_StateFolder / f, skiprows=1, index_col=["PartyAb", "StateAb"]
    )
    state_tmp["DivisionNm"] = "State"
    state_tmp.set_index("DivisionNm", append=True, inplace=True)
    state_tmp = state_tmp["TotalPercentage"]

    # And the National ones
    national_tmp = pd.read_csv(
        fp.correlations_NationalFolder / f, skiprows=1, index_col=["PartyAb"]
    )
    national_tmp["DivisionNm"] = "NAT"
    national_tmp["StateAb"] = "NAT"
    national_tmp.set_index(["StateAb", "DivisionNm"], append=True, inplace=True)
    national_tmp = national_tmp["TotalPercentage"]

    year_data = pd.concat((first_pref_percentage, state_tmp, national_tmp))

    year_data.name = year

    return year_data


files = [
    "2022.csv",
    "2019.csv",
    "2016.csv",
    "2013.csv",
    "2010.csv",
    "2007.csv",
    "2004.csv",
]

df = make_year_data(files[0])


master = pd.DataFrame(index=df.index)

for f in files:
    year = int(f.strip(".csv"))
    print(year)

    tmp = pd.read_csv(fp.correlations_DivisionFolder / f, skiprows=1)
    party_votes = tmp.groupby(["PartyAb", "StateAb", "DivisionNm"])["TotalVotes"].sum()
    total_votes = tmp.groupby(["StateAb", "DivisionNm"])["TotalVotes"].sum()

    first_pref_percentage = 100 * (party_votes / total_votes)

    # Add in the state results
    state_tmp = pd.read_csv(
        fp.correlations_StateFolder / f, skiprows=1, index_col=["PartyAb", "StateAb"]
    )
    state_tmp["DivisionNm"] = "State"
    state_tmp.set_index("DivisionNm", append=True, inplace=True)
    state_tmp = state_tmp["TotalPercentage"]

    # And the National ones
    national_tmp = pd.read_csv(
        fp.correlations_NationalFolder / f, skiprows=1, index_col=["PartyAb"]
    )
    national_tmp["DivisionNm"] = "NAT"
    national_tmp["StateAb"] = "NAT"
    national_tmp.set_index(["StateAb", "DivisionNm"], append=True, inplace=True)
    national_tmp = national_tmp["TotalPercentage"]

    year_data = pd.concat((first_pref_percentage, state_tmp, national_tmp))

    year_data.name = year

    master = pd.merge(master, year_data, left_index=True, right_index=True, how="outer")


master = master.sort_index()

# Fix the liberal/national coalition values
# Sum the votes of the liberals and the nationals
liberal_party_voteshare = master.loc[idx["LP", :, "State"]].fillna(0)
national_party_voteshare = master.loc[idx["NP", :, "State"]].fillna(0)
LNP_voteshare = (
    national_party_voteshare.reindex(liberal_party_voteshare.index).fillna(0)
    + liberal_party_voteshare
)
# Then fix Queensland after 2010
LNP_voteshare_queensland = (
    master.loc["LNP", :, "State"].reindex(LNP_voteshare.index).fillna(0)
)
LNP_voteshare = LNP_voteshare + LNP_voteshare_queensland

# Now add the national results
LNP_national = (
    master.loc[idx["LNP", "NAT"]].fillna(0)
    + master.loc[idx["NP", "NAT"]].fillna(0)
    + master.loc[idx["LP", "NAT"]].fillna(0)
)
LNP_voteshare = pd.concat((LNP_voteshare, LNP_national))
LNP_voteshare["DivisionNm"] = ["State"] * 7 + ["NAT"]
LNP_voteshare = LNP_voteshare.reset_index(names=["StateAb"])
LNP_voteshare["PartyAb"] = "LNP"


# Now reindex it
LNP_voteshare = LNP_voteshare.set_index(["PartyAb", "StateAb", "DivisionNm"])

national_state_correlations = master.loc[idx[["ALP", "GRN"], :, ["NAT", "State"]]]
national_state_correlations = pd.concat((national_state_correlations, LNP_voteshare))

national_state_correlations = national_state_correlations.reset_index()
national_state_correlations["UniqueIdentifier"] = (
    national_state_correlations["PartyAb"]
    + "_"
    + national_state_correlations["StateAb"]
)
national_state_correlations.to_csv(
    fp.three_party_voteshare_data,
    index=False,
)
