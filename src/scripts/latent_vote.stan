data{
    int<lower=1> N_polls;
    int<lower=1> N_pollsters;

    // Number of days since 2022 election
    array[N_polls] int N_days;
    // Integer corresponding to each pollster
    array[N_polls] int PollName_Encoded;

    // Day we want to end our model on
    int<lower=max(N_days)> prediction_date;

    // Sample size for each election
    array[N_polls] int survey_size;
    // Result of the poll
    array[N_polls] real poll_result;

    // Result of the election in 2022
    real election_result_2022;

    // Additional variance to *add* to each poll
    real<lower=0> additional_variance;

}

transformed data {
   vector[N_polls] poll_var;

   for (i in 1:N_polls) {
        poll_var[i] = poll_result[i] * (1 - poll_result[i]) / survey_size[i];
   }
}

parameters {
  vector[prediction_date] mu;
  vector[N_pollsters] house_effects;
  real<lower=0> sigma;
  real<lower=1> nu;
  real<lower=0> time_variation_factor;
  real<lower=1> time_variation_day_scale;

}

transformed parameters {

  vector[prediction_date] time_variation_sigma;

  for (t in 1:prediction_date) {
    time_variation_sigma[t] = time_variation_factor * exp(-t / time_variation_day_scale);
  }
}



model{

    // Prior on the state space variance
    sigma ~ normal(0.0, 0.1);

    // "Measurement" on election day 2022 with a very small error
    mu[1] ~ normal(election_result_2022, 0.0001);

    // Prior on the house effects
    house_effects ~ normal(0, 0.05);

    nu ~ gamma(2,0.1);

    time_variation_factor ~ normal(0.0, 0.01);
    time_variation_day_scale ~ normal(0, 30);

    // state model
    mu[2:prediction_date] ~ student_t(nu, mu[1:(prediction_date- 1)], sigma + time_variation_sigma[1:(prediction_date -1)]);

    // add in the polls on each day
    poll_result ~ normal(mu[N_days] + house_effects[PollName_Encoded], sqrt(poll_var + additional_variance));
}