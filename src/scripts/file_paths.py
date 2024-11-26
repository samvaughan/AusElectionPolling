from pathlib import Path


correlations_DivisionFolder = Path("src/data/Correlations/ByDivision")
correlations_StateFolder = Path("src/data/Correlations/ByState")
correlations_NationalFolder = Path("src/data/Correlations/National")
correlations_processed = Path("src/data/Correlations/Processed")

# Files
latest_poll_data = Path("src/data/Polls/poll_data_latest.csv")
three_party_voteshare_data = Path(
    correlations_processed / "National_State_correlations_ALP_LNP_GRN.csv"
)


def get_filenames(date, first_pref_or_2pp):

    assert first_pref_or_2pp in [
        "first_preference",
        "2pp",
    ], "Must be one of first_preference or 2pp"

    # Main folder for inference data
    master_folder_inference = Path(f"src/data/InferenceData/{date}/")
    master_folder_inference.mkdir(exist_ok=True)

    # Main folder for model data
    master_folder_Model = Path(f"src/data/ModelData/{date}/")
    master_folder_Model.mkdir(exist_ok=True)

    netcdf_filename = master_folder_inference / f"{first_pref_or_2pp}_{date}"
    model_data_filename = master_folder_Model / f"{first_pref_or_2pp}_{date}.pkl"
    model_df_filename = master_folder_Model / f"{first_pref_or_2pp}_{date}.csv"
    election_data_filename = master_folder_Model / "election_data.csv"

    return (
        netcdf_filename,
        model_data_filename,
        model_df_filename,
        election_data_filename,
    )


def get_filenames_single_party(party_short_name, date):

    # Main folder for inference data
    master_folder_inference = Path(f"src/data/InferenceData/{date}/")
    master_folder_inference.mkdir(exist_ok=True)

    # Main folder for model data
    master_folder_Model = Path(f"src/data/ModelData/{date}/")
    master_folder_Model.mkdir(exist_ok=True)

    netcdf_filename = master_folder_inference / f"{party_short_name}_{date}"
    model_data_filename = master_folder_Model / f"{party_short_name}_{date}.pkl"
    model_df_filename = master_folder_Model / f"{party_short_name}_{date}.csv"
    election_data_filename = master_folder_Model / "election_data.csv"

    return (
        netcdf_filename,
        model_data_filename,
        model_df_filename,
        election_data_filename,
    )
