import numpy as np
import matplotlib.pyplot as plt
import cmdstanpy
import arviz as az

seed = 123
rng = np.random.default_rng(seed)

sigma = 0.001

# time_variation_factor = 0.005
# time_variation_day_scale = 15

N_polls = 200
N_pollsters = 9
prediction_date = 900

additional_variance = 0

election_value = 0.52

mu = np.zeros(prediction_date)

# time_factor = time_variation_factor * np.exp(
#     -np.arange(prediction_date) / time_variation_day_scale
# )
deltas = rng.normal(loc=0, scale=sigma, size=prediction_date)
deltas[0] = 0

mu = deltas.cumsum() + election_value

poll_dates = rng.choice(np.arange(prediction_date), size=N_polls)
survey_size = rng.normal(loc=1000, scale=100, size=N_polls).astype(int)
poll_errors = np.sqrt((mu[poll_dates] * (1 - mu[poll_dates])) / survey_size)

# Make the encoding for the poll effects
poll_name = rng.choice(np.arange(N_pollsters), size=N_polls)

house_effects = rng.normal(loc=0, scale=0.03, size=N_pollsters)

poll_results = (
    rng.normal(loc=mu[poll_dates], scale=poll_errors) + house_effects[poll_name]
)

markers = np.array(["o", "s", "*", "+", "v", "^", "P", "X", "H"])

# Now lay out all of the data
data = dict(
    N_polls=N_polls,
    N_pollsters=N_pollsters,
    N_days=poll_dates,
    PollName_Encoded=poll_name + 1,
    prediction_date=prediction_date,
    poll_error=poll_errors,
    poll_result=poll_results,
    election_result_2022=election_value,
    additional_variance=additional_variance,
)

# Plot the simulated data
fig, ax = plt.subplots()
ax.plot(mu, c="r", linewidth=3.0)
for i in np.arange(len(poll_dates)):
    ax.errorbar(
        poll_dates[i],
        data["poll_result"][i],
        yerr=data["poll_error"][i],
        marker=markers[poll_name][i],
        linestyle="None",
        markersize=5,
        c="r",
        alpha=0.3,
    )

model = cmdstanpy.CmdStanModel(stan_file="src/scripts/latent_vote.stan")

fit = model.sample(data=data, adapt_delta=0.95)

# Turn this into inference data
coords = {
    "pollster": np.arange(N_pollsters),
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


# Plot the overall vote fractions
az.plot_forest(
    trace, var_names=["house_effects"], transform=lambda x: 100 * x, combined=True
)
for j, (y_tick, frac_j) in enumerate(
    zip(plt.gca().get_yticks(), reversed(house_effects * 100))
):
    plt.vlines(
        frac_j, ymin=y_tick - 0.45, ymax=y_tick + 0.45, color="black", linestyle="--"
    )
