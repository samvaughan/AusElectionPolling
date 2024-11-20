data{
    int<lower=1> N_polls;
    int<lower=1> N_pollsters;
    int<lower=1> N_parties;

    // Date of each poll as measured from 2022 election
    array[N_polls] int N_days;
    // Integer corresponding to each pollster
    array[N_polls] int PollName_Encoded;

    // Day we want to end our model on
    int<lower=max(N_days)> prediction_date;

    // Sample size for each election
    array[N_polls] row_vector[N_parties] poll_variance;

    // Result of the poll
    array[N_polls] vector[N_parties] poll_result;

    // Result of the election in 2022
    row_vector[N_parties] election_result_2022;

    // Incumbant flag
    row_vector[N_parties] incumbent;

    // Additional variance to *add* to each poll
    real<lower=1> inflator;

}

parameters {
  array[prediction_date] row_vector[N_parties] mu_raw;
  array[N_pollsters] row_vector[N_parties] house_effects;
  row_vector<lower=0>[N_parties] sigma;
  //real<lower=1> nu;
  //real<lower=0> time_variation_factor;
  //real<lower=1> time_variation_day_scale;

}

transformed parameters {

  array[prediction_date] row_vector[N_parties] mu;
  // vector[prediction_date -1] time_variation_sigma;

  // for (t in 1:prediction_date-1) {
  //   time_variation_sigma[t] = time_variation_factor * exp(-t / 20.0);
  // }

  mu[1] = election_result_2022;
  for (t in 2:prediction_date){
    mu[t] = mu[t-1] + (sigma .* mu_raw[t-1]);
  }

}


model{

    // Prior on the state space variance
    sigma ~ normal(0.0, 0.03);

    for (p in 1:N_pollsters){
      house_effects[p] ~ normal(0, 0.02);
    }
    // //nu ~ gamma(2,0.1);
    //time_variation_factor ~ normal(0.0, 10);
    // time_variation_day_scale ~ normal(0, 30);

    // state model
    // for (t in 2:prediction_date){
    //   mu[t] ~ normal(mu[t-1], sigma);
    // }

    for (t in 1:prediction_date){
      mu_raw[t] ~ std_normal();
    }

    //Election measurement
    election_result_2022 ~ normal(mu[1], 0.001);

    // Poll measurement
    for (i in 1:N_polls){
      // add in the polls on each day
      poll_result[i] ~ normal(mu[N_days[i]] + house_effects[PollName_Encoded[i]], inflator * sqrt(poll_variance[i]));
    }


}