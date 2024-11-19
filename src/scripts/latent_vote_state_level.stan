data{
    int<lower=1> N_polls;
    int<lower=1> N_pollsters;
    int<lower=1> N_states;

    // Number of days since 2022 election
    array[N_polls] int N_days;
    // Integer corresponding to each pollster
    array[N_polls] int PollName_Encoded;

    // Poll level: 1 = national, 2 = Victoria
    array[N_polls] int PollLevel;

    // Day we want to end our model on
    int<lower=max(N_days)> prediction_date;

    // Sample size for each election
    array[N_polls] int survey_size;
    // Result of the poll
    array[N_polls] real poll_result;

    // Result of the elections at the start and finish
    real election_result_start;
    real state_election_result_start;

    matrix[N_states + 1, N_states + 1] Sigma;

    // Additional variance to *add* to each poll
    //real<lower=0> additional_variance;

}

transformed data {
   vector[N_polls] poll_var;

   for (i in 1:N_polls) {
        poll_var[i] = poll_result[i] * (1 - poll_result[i]) / survey_size[i];
   }
}

parameters {
  matrix[prediction_date, N_states + 1] mu;
  vector[N_pollsters] house_effects;
  real<lower=0> sigma;
  real<lower=1> nu;
  real<lower=0> additional_variance;
  real swing;
}

model{

    // Prior on the state space variance
    sigma ~ normal(0.005, 0.005);
    additional_variance ~ normal(0, 0.05);

    // "Measurement" on election day 2022 with a very small error
    mu[1, 1] ~ normal(election_result_start, 0.0001);
    mu[1, 2] ~ normal(state_election_result_start, 0.0001);


    // Prior on the house effects
    house_effects ~ normal(0, 0.05);

    nu ~ gamma(2,0.1);

    swing ~ normal(0, 0.02);


    // state model
    // // This is saying that the national and state-level movements are uncorrelated, which is obviously not true! To improve...
    // mu[2:prediction_date, 1] ~ student_t(nu, mu[1:(prediction_date- 1), 1], sigma);
    // mu[2:prediction_date, 2] ~ student_t(nu, mu[1:(prediction_date- 1), 1] + swing[1:(prediction_date) - 1, 1], sigma);
    to_row_vector(mu[2:prediction_date]) ~ multi_normal(to_row_vector(mu[1:(prediction_date) - 1]) + [0, swing], Sigma);

    // add in the polls on each day
    poll_result ~ normal(to_vector(mu[N_days, PollLevel]) + house_effects[PollName_Encoded], sqrt(poll_var + additional_variance));

}