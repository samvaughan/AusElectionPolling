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

    // Result of the elections at the start and finish
    real election_result_start;
    real election_result_finish;

    // Additional variance to *add* to each poll
    //real<lower=0> additional_variance;

}

parameters {
  vector[prediction_date-1] mu_raw;
  vector[N_pollsters] house_effects;
  real<lower=0> sigma;
  // real<lower=1> nu;
  // real<lower=0> additional_variance;
}

transformed parameters {

  vector[prediction_date] mu;

  mu[1] = election_result_start;
  for (t in 2:prediction_date){
    mu[t] = mu[t-1] + (sigma) * mu_raw[t-1];
  }

}

model{

    // Prior on the state space variance
    sigma ~ normal(0.005, 0.005);
    // additional_variance ~ normal(0, 0.05);

    // Prior on the house effects
    house_effects ~ normal(0, 0.05);
    //nu ~ gamma(2,0.1);

    // state model
    mu_raw ~ std_normal();

    // "Measurement" on election day 2022 with a very small error
    election_result_start ~ normal(mu[1], 0.0001);
    election_result_finish ~ normal(mu[prediction_date], 0.0001);

    // add in the polls on each day
    poll_result ~ normal(mu[N_days] + house_effects[PollName_Encoded], poll_error);
}