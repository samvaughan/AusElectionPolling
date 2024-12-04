functions {
  /*
    Extract lower-triangular from mat to vector.
    Excludes diagonal.
    @param mat Matrix.
    @return vector
   */
  vector lower_tri(matrix mat) {
    int d = rows(mat);
    int lower_tri_d = d * (d - 1) / 2;
    vector[lower_tri_d] lower_mat;
    int count = 1;
    for(r in 2:d) {
      for(c in 1:(r - 1)) {
        lower_mat[count] = mat[r,c];
        count += 1;
      }
    }
    return(lower_mat);

  }
  /*
    Proportional log-density for correlation matrix.
    @param cor Matrix. PSD, symmetric correlation matrix.
    @param point_mu_lower Vector. D(D-1)/2 length. Locations of each [unique] element's prior.
    @param point_scale_lower Vector. D(D-1)/2 length. Scales of each [unique] element's prior.
    @return log density.
   */
  real lkj_corr_point_lower_tri_lpdf(matrix cor, vector point_mu_lower, vector point_scale_lower) {
    real lpdf = lkj_corr_lpdf(cor | 1) + normal_lpdf(lower_tri(cor) | point_mu_lower, point_scale_lower);
    return(lpdf);
  }
  /*
    Same as lkj_corr_point_lower_tri_lpdf, but takes matrices of prior locations and scales.
   */
  real lkj_corr_point_lpdf(matrix cor, matrix point_mu, matrix point_scale) {
    int d = rows(cor);
    int lower_tri_d = d * (d - 1) / 2;
    vector[lower_tri_d] point_mu_lower = lower_tri(point_mu);
    vector[lower_tri_d] point_scale_lower = lower_tri(point_scale);
    real out = lkj_corr_point_lower_tri_lpdf(cor| point_mu_lower, point_scale_lower);
    return(out);

  }
  /*
    Same as lkj_corr_point_lower_tri_lpdf, but takes cholesky-factor of correlation matrix.
   */
  real lkj_corr_cholesky_point_lower_tri_lpdf(matrix cor_L, vector point_mu_lower, vector point_scale_lower) {
    real lpdf = lkj_corr_cholesky_lpdf(cor_L | 1);
    int d = rows(cor_L);
    matrix[d,d] cor = multiply_lower_tri_self_transpose(cor_L);
    lpdf += normal_lpdf(lower_tri(cor) | point_mu_lower, point_scale_lower);
    return(lpdf);
  }
}


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
    //cholesky_factor_corr[N_parties] cholesky_matrix;
    cholesky_factor_corr[N_parties] cholesky_matrix_loc;
    matrix[N_parties, N_parties] cholesky_matrix_scale;

    // Result of the election in 2022
    array[N_parties] real election_result;
    array[N_parties] int election_result_index;

    // // Incumbant flag
    // row_vector[N_parties] incumbent;

    // Additional variance to *add* to each poll
    real<lower=1> inflator;

}

transformed data {
  vector[N_parties * (N_parties - 1) / 2] cor_guess_lower;
  vector<lower=0>[N_parties * (N_parties - 1) / 2] cor_sd_lower;

  cor_guess_lower = lower_tri(cholesky_matrix_loc);
  cor_sd_lower = lower_tri(cholesky_matrix_scale);
}

parameters {
  matrix[prediction_date-1, N_parties] mu_raw;
  array[N_pollsters] row_vector[N_parties] house_effects;
  vector<lower=0>[N_parties] sigma;
  cholesky_factor_corr[N_parties] cor_L;
}

transformed parameters {

  matrix[prediction_date, N_parties] mu;
  matrix[prediction_date-1, N_parties] Delta;
  matrix[N_parties, N_parties] Sigma;

  Sigma = diag_pre_multiply(sigma, cor_L);


  for (p in 1:N_parties){
    mu[1, election_result_index[p]] = election_result[election_result_index[p]];
    //mu[2:,election_result_index[p]] = cumulative_sum(Delta[:,election_result_index[p]]) + election_result[election_result_index[p]];
  }

  for (t in 1:prediction_date - 1){
    // Delta[t] = mu_raw[t] * Sigma;
    mu[t+1] = mu[t] + mu_raw[t] * Sigma;
  }


}


model{

    // Prior on the state space variance
    sigma ~ normal(0.0, 0.03);

    for (p in 1:N_pollsters){
      house_effects[p] ~ normal(0, 0.02);
    }

    //Prior on the covariance matrix
    cor_L ~ lkj_corr_cholesky_point_lower_tri(cor_guess_lower, cor_sd_lower);
    for (t in 1:prediction_date-1){
      mu_raw[t] ~ std_normal();
    }

    //Election measurement
    for (p in 1:N_parties){
      election_result[election_result_index[p]] ~ normal(mu[1, election_result_index[p]], 0.001);
    }

    // Poll measurement
    for (i in 1:N_polls){
      poll_result[i] ~ normal(mu[N_days[i], party_index[i]] + house_effects[PollName_Encoded[i], party_index[i]], inflator * sqrt(poll_variance[i]));
    }


}

generated quantities {
    matrix[prediction_date, N_parties] nu;

    for (p in 1:N_parties){
        nu[1, election_result_index[p]] = election_result[election_result_index[p]];
    }

    for (t in 2:prediction_date){
          nu[t,:] = to_row_vector(multi_normal_cholesky_rng(mu[t-1, :], Sigma));
      }
}