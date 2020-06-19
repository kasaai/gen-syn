make_analysis_fn <- function(batch_size, training_fn, pre_process_fn, epochs,
                             post_process_fn) {
  function(split) {
    train <- analysis(split)
    test <- assessment(split)
    synthesizer <- ctgan(batch_size = batch_size)
    train_syn <- train %>% 
      pre_process_fn()
    synthesizer %>% 
      fit(train_syn, log_frequency = FALSE, epochs = epochs)
    syn <- synthesizer %>% 
      ctgan_sample(n = nrow(train)) %>% 
      post_process_fn()
    
    glm_syn <- training_fn(syn)
    glm_train <- training_fn(train)
    
    coef_glm_syn <- broom::tidy(glm_syn)
    coef_glm_train <- broom::tidy(glm_train)
    
    preds_syn <- predict(glm_syn, test, type = "response")
    preds_train <- predict(glm_train, test, type = "response")
    actual_response <- test$num_claims
    
    rmse_syn <- compute_rmse(preds_syn, actual_response)
    rmse_train <- compute_rmse(preds_train, actual_response)
    
    list(rmse_syn = rmse_syn, rmse_train = rmse_train,
         coef_glm_syn = coef_glm_syn, coef_glm_train = coef_glm_train,
         real_data = train, synthesized_data = syn)
  }
}

run_cv <- function(data, batch_size, training_fn, pre_process_fn,
                   epochs, post_process_fn, n_folds) {
  analysis_fn <- make_analysis_fn(batch_size, training_fn, pre_process_fn, epochs,
                                  post_process_fn)
  folds <- rsample::vfold_cv(modeling_data, v = n_folds)
  result <- map(folds$splits, analysis_fn)
  
  result
}