data{
    int<lower=1> N_polls;
    int<lower=1> N_pollsters;
    int<lower=1> N_parties;

    // Number of days since 2022 election
    array[N_polls] int N_days;
    // Integer corresponding to each pollster
    array[N_polls] int PollName_Encoded;

    // Day we want to end our model on
    int<lower=max(N_days)> prediction_date;

    // Sample size for each election
    array[N_polls] int survey_size;
    // Result of the poll
    array[N_polls] vector[N_parties] poll_result;

    // Result of the election in 2022
    vector[N_parties] election_result_2022;

    // Additional variance to *add* to each poll
    real<lower=0> additional_variance;

}

transformed data {
   array[N_polls] vector[N_parties] poll_var;

   for (i in 1:N_polls) {
    for (j in 1:N_parties){
        poll_var[i, j] = poll_result[i, j] * (1 - poll_result[i, j]) / survey_size[j];
      }
   }
}

parameters {
  array[prediction_date] row_vector[N_parties] mu;
  array[N_pollsters] row_vector[N_parties] house_effects;
  vector<lower=0>[N_parties] sigma;
  //real<lower=1> nu;
  real<lower=0> time_variation_factor;
  real<lower=1> time_variation_day_scale;

}

transformed parameters {

  vector[prediction_date] time_variation_sigma;
  // array[N_polls] row_vector[N_parties] normal_loc;
  // array[N_polls] matrix[N_parties, N_parties] normal_Sigma;


  for (t in 1:prediction_date) {
    time_variation_sigma[t] = time_variation_factor * exp(-t / time_variation_day_scale);
  }

  // for (t in 1:N_polls){
  //   normal_loc[t] = mu[N_days[t]] + house_effects[PollName_Encoded[t]];
  //   normal_Sigma[t] = sqrt(poll_var[t] + additional_variance);
  // }



}



model{

    // Prior on the state space variance
    sigma ~ normal(0.0, 0.1);

    // "Measurement" on election day 2022 with a very small error

    mu[1] ~ normal(election_result_2022, 0.0001);

    for (p in 1:N_pollsters){
      house_effects[p] ~ normal(0, 0.05);
    }
    //nu ~ gamma(2,0.1);
    time_variation_factor ~ normal(0.0, 0.01);
    time_variation_day_scale ~ normal(0, 30);

    // state model
    for (t in 2:prediction_date){
      mu[t] ~ normal(mu[t-1], (sigma + time_variation_sigma[t]));
    }

    for (t in 1:N_polls){
      // add in the polls on each day
      poll_result[t] ~ normal(mu[t] + house_effects[PollName_Encoded[t]], sqrt(poll_var[t] + additional_variance));
    }


}