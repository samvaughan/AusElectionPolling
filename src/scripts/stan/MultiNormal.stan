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
    array[N_polls] real poll_variance;

    // Result of the poll
    array[N_polls] real poll_result;

    array[N_polls] int party_index;

    //corr_matrix[N_parties] correlation_matrix;
    cholesky_factor_corr[N_parties] cholesky_matrix;

    // Result of the election in 2022
    row_vector[N_parties] election_result_2022;

    // // Incumbant flag
    // row_vector[N_parties] incumbent;

    // Additional variance to *add* to each poll
    real<lower=1> inflator;

}

parameters {
  matrix[prediction_date-1, N_parties] mu_raw;
  array[N_pollsters] row_vector[N_parties] house_effects;
  vector<lower=0>[N_parties] sigma;
}

transformed parameters {

  matrix[prediction_date, N_parties] mu;
  matrix[prediction_date-1, N_parties] Delta;
  matrix[N_parties, N_parties] Sigma;

  Sigma = diag_pre_multiply(sigma, cholesky_matrix);

  for (t in 1:prediction_date - 1){
    Delta[t] = mu_raw[t] * Sigma;
  }

  mu[1] = election_result_2022;
  for (p in 1:N_parties){
    mu[2:,p] = cumulative_sum(Delta[:,p]) + election_result_2022[p];
  }

}


model{

    // Prior on the state space variance
    sigma ~ normal(0.0, 0.03);

    for (p in 1:N_pollsters){
      house_effects[p] ~ normal(0, 0.02);
    }

    //L ~ lkj_corr_cholesky(2);
    for (t in 1:prediction_date-1){
      mu_raw[t] ~ std_normal();
    }

    //Election measurement
    election_result_2022 ~ normal(mu[1], 0.001);
    // Poll measurement
    for (i in 1:N_polls){
      poll_result[i] ~ normal(mu[N_days[i], party_index[i]] + house_effects[PollName_Encoded[i], party_index[i]], inflator * sqrt(poll_variance[i]));
    }


}