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
    array[N_polls] real poll_error;
    // Result of the poll
    array[N_polls] real poll_result;

    // Result of the election in 2022
    real election_result_2022;

    int incumbent;

    // Additional variance to *add* to each poll
    real<lower=0> additional_variance;

}


parameters {
  vector[prediction_date-1] mu_raw;
  vector[N_pollsters] house_effects;
  real<lower=0> sigma;
  //real<lower=1> time_variation_day_scale;
  real<lower=1> time_variation_factor;
}

transformed parameters {

  vector[prediction_date] mu;
  vector[prediction_date - 1] time_variation_sigma;

  for (t in 1:prediction_date - 1) {
    time_variation_sigma[t] = (exp(-t / 20.0));
  }

  mu[1] = election_result_2022;
  for (t in 2:prediction_date){
    mu[t] = mu[t-1] + (sigma * (1 + incumbent * time_variation_factor * time_variation_sigma[t-1])) * mu_raw[t-1];
  }

}



model{

    real delta;
    // Prior on the state space variance
    sigma ~ normal(0, 0.01);

    // Prior on the house effects
    house_effects ~ normal(0, 0.05);

    time_variation_factor ~ normal(0, 3);
    //time_variation_day_scale ~ normal(0, 50);
    mu_raw ~ std_normal();

    // "Measurement" on election day 2022 with a very small error
    election_result_2022 ~ normal(mu[1], 0.001);

    // add in the polls on each day
    poll_result ~ normal(mu[N_days] + house_effects[PollName_Encoded], poll_error);
}