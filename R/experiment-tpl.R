library(cellar)
library(ctgan)
library(tidyverse)
library(rsample)

source("R/util.R")

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

training_fn <- function(training_data) {
  f <- num_claims ~ vehicle_power + vehicle_age + driver_age +
    bonus_malus + vehicle_brand + vehicle_gas + density +
    region + area - 1
  glm(f, family = poisson(), data = training_data, offset = log(exposure))
}

compute_rmse <- function(preds, actuals) {
  sqrt(sum((preds - actuals)^2) / length(preds))
}

pre_process <- function(x) {
  x %>% 
    sample_n(100000) %>% 
    mutate_at("num_claims", as.character)
}

post_process <- function(x) {
  x %>% 
    mutate(
      num_claims = as.integer(num_claims),
      exposure = pmin(pmax(1 / 365, exposure), 1)
    )
}

result <- run_cv(
  data = modeling_data,
  batch_size = 50000,
  training_fn = training_fn,
  pre_process_fn = pre_process,
  post_process_fn = post_process,
  epochs = 100,
  n_folds = 10
)