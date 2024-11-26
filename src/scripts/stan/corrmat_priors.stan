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
    vector[lower_tri_d] lower;
    int count = 1;
    for(r in 2:d) {
      for(c in 1:(r - 1)) {
	lower[count] = mat[r,c];
	count += 1;
      }
    }
    return(lower);

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

data {
  int N; // # observations
  int D; // # variables
  vector[D] y[N]; // Data
  matrix[D,D] cor_guess; // Prior correlation location guesses.
  matrix[D,D] cor_sd; // Prior correlation uncertainties/scales.
  int just_lkj; // Whether to just use LKJ, or use defined lpdfs.
}

transformed data {
  int d_unique = D*(D-1)/2;
  vector[d_unique] cor_guess_lower = lower_tri(cor_guess);
  vector[d_unique] cor_sd_lower = lower_tri(cor_sd);
}

parameters {
  vector[D] mu;
  vector<lower=0>[D] sigma;
  cholesky_factor_corr[D] cor_L;
}

model {
  mu ~ std_normal();
  sigma ~ std_normal();
  if(just_lkj) {
    cor_L ~ lkj_corr_cholesky(1);
  } else {
    cor_L ~ lkj_corr_cholesky_point_lower_tri(cor_guess_lower, cor_sd_lower);
  }

  y ~ multi_normal_cholesky(mu, diag_pre_multiply(sigma, cor_L));
}

generated quantities {
  matrix[D,D] cor = multiply_lower_tri_self_transpose(cor_L);
}
