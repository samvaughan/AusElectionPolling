def get_filenames(party_short_name, date):
    netcdf_filename = f"src/data/InferenceData/{party_short_name}_{date}"
    model_data_filename = f"src/data/ModelData/{party_short_name}_{date}.pkl"
    model_df_filename = f"src/data/ModelData/{party_short_name}_{date}.csv"
    election_data_filename = "src/data/ModelData/election_data.csv"

    return (
        netcdf_filename,
        model_data_filename,
        model_df_filename,
        election_data_filename,
    )
