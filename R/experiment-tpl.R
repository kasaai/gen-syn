library(cellar)
library(ctgan)
library(tidyverse)
library(rsample)

# download data if needed
policies <- cellar_pull("fr_tpl2_policies")

# simple transformations
modeling_data <- policies %>% 
  mutate(
    num_claims = pmin(num_claims, 4),
    exposure = pmin(exposure, 1),
    vehicle_power = ifelse(as.integer(vehicle_power) > 9, "9", vehicle_power),
    vehicle_age = case_when(
      vehicle_age < 1 ~ "< 1",
      vehicle_age <= 10 ~ "[1, 10]",
      TRUE ~ "> 10"
    ),
    driver_age = case_when(
      driver_age < 21 ~ "< 21",
      driver_age < 26 ~ "[21, 26)",
      driver_age < 31 ~ "[26, 31)",
      driver_age < 41 ~ "[41, 51)",
      driver_age < 71 ~ "[51, 71)", 
      TRUE ~ ">= 71"
    ),
    bonus_malus = pmin(bonus_malus, 150),
    density = log(density)#,
  ) %>% 
  select(-policy_id)

create_glm <- function(training_data) {
  f <- num_claims ~ vehicle_power + vehicle_age + driver_age +
    bonus_malus + vehicle_brand + vehicle_gas + density +
    region + area
  glm(f, family = poisson(), data = training_data, offset = log(exposure))
}

compute_rmse <- function(preds, actuals) {
  sqrt(sum((preds - actuals)^2) / length(preds))
}

analyze_synthesis <- function(split) {
  train <- analysis(split)
  test <- assessment(split)
  synthesizer <- ctgan()
  train_syn <- train %>% 
    sample_n(100000) %>% 
    mutate_at("num_claims", as.character)
  synthesizer %>% 
    fit(train_syn,batch_size = 10000, epochs = 300)
  syn <- synthesizer %>% 
    ctgan_sample(n = nrow(train), batch_size = 10000) %>% 
    mutate(num_claims = as.integer(num_claims),
           exposure = pmin(pmax(1/365, exposure), 1))
  
  glm_syn <- create_glm(syn)
  glm_train <- create_glm(train)
  
  preds_syn <- predict(glm_syn, test, type = "response")
  preds_train <- predict(glm_train, test, type = "response")
  actual_response <- test$num_claims
  
  rmse_syn <- compute_rmse(preds_syn, actual_response)
  rmse_train <- compute_rmse(preds_train, actual_response)
  
  list(rmse_syn = rmse_syn, rmse_train = rmse_train)
}

folds <- rsample::vfold_cv(modeling_data, v = 10)
result <- map(folds$splits, analyze_synthesis)
rmses <- result %>% transpose() %>% map(flatten_dbl)
rmses

# saveRDS(rmses, "tpl.rds")