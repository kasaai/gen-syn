library(cellar)
library(ctgan)
library(tidyverse)
library(rsample)

# download data if needed
lapse_study <- cellar_pull("lapse_study")
glimpse(lapse_study)

issue_age_mapping <- tribble(
  ~ age_band, ~ avg_issue_age,
  "0-19",     10,
  "20-29",    25,
  "30-39",    35,
  "40-49",    45,
  "50-59",    55,
  "60-69",    65,
  "70+",      75,
)

# Map levels to midpoint of bands per SOA2015
premium_jump_ratio_mapping <- tribble(
  ~ premium_jump_ratio_band, ~ avg_premium_jump_ratio,
  "A.  1.01 - 2.00",    1.5,
  "B.  2.01 - 3.00",    2.5,
  "C.  3.01 - 4.00",    3.5,
  "D.  4.01 - 5.00",    4.5,
  "E.  5.01 - 6.00",    5.5,
  "F.  6.01 - 7.00",    6.5,
  "G.  7.01 - 8.00",    7.5,
  "H.  8.01 - 9.00",    8.5,
  "I.  9.01 - 10.00",   9.5,
  "J. 10.01 - 11.00",   10.5,
  "K. 11.01 - 12.00",   11.5,
  "L. 12.01 - 13.00",   12.5,
  "M. 13.01 - 14.00",   13.5,
  "N. 14.01 - 15.00",   14.5,
  "O. 15.01 - 16.00",   15.5,
  "P. 16.01 - 17.00",   16.5,
  "Q. 17.01 - 18.00",   17.5,
  "R. 18.01 - 19.00",   18.5,
  "S. 19.01 - 20.00",   19.5,
  "T. 20.01 - 21.00",   20.5,
  "U. 21.01 - 22.00",   21.5,
  "V. 22.01 - 23.00",   22.5,
  "W. 23.01 - 24.00",   23.5,
  # SOA 2014 p. 18
  "X. 24.01 AND UP",    27.9
)

risk_class_mapping <- tribble(
  ~ risk_class,    ~ risk_class_mapped,
  "Super-Pref NS", "BCNS",
  "Non-Pref NS",   "NS",
  "Pref Resid NS", "NS",
  "Pref Best NS",  "BCNS",
  "Undiff SM",     "SM",
  "Non-Pref SM",   "SM",
  "Pref SM",       "SM",
  "Undiff NS",     "NS",
  "Agg/Unknown",   "NS"
)

premium_mode_mapping <- tribble(
  ~ premium_mode,   ~ premium_mode_mapped,
  "1. Annual", "Annual",
  "2. Semiannual", "Semiannual/Quarterly",
  "3. Quarterly", "Semiannual/Quarterly",
  "4. Monthly", "Monthly",
  "5. Biweekly", "Other/Unknown",
  "6. Unknown/Other", "Other/Unknown"
)

modeling_data <- lapse_study %>% 
  filter(duration %in% c("10", "11", "12")) %>%
  # Keep only Premium Jump to ART
  filter(post_level_premium_structure == "1. Premium Jump to ART") %>%
  # Keep only known premium jump ratios
  filter(premium_jump_ratio != "Y. Unknown") %>%
  # Remove empty exposures.
  filter(exposure_count > 0, exposure_amount > 0) %>%
  # Join with issue age mapping
  left_join(issue_age_mapping, by = c(issue_age = "age_band")) %>%
  # risk class mapping for SOA 2015
  left_join(risk_class_mapping, by = "risk_class") %>%
  # premium mode mapping for SOA 2015
  left_join(premium_mode_mapping, by = "premium_mode") %>%
  # Join with premium jump ratio mapping
  left_join(premium_jump_ratio_mapping, by = c(premium_jump_ratio = "premium_jump_ratio_band")) %>% 
  select(lapse_count, risk_class_mapped, face_amount, premium_mode_mapped,
         avg_issue_age, avg_premium_jump_ratio, duration, exposure_count)

create_glm <- function(training_data) {
  f <- lapse_count ~
    risk_class_mapped + face_amount +
    avg_issue_age + avg_premium_jump_ratio + duration
  
  glm(f, family = poisson(), data = training_data, offset = log(exposure_count))
}

compute_rmse <- function(preds, actuals) {
  sqrt(sum((preds - actuals)^2) / length(preds))
}

analyze_synthesis <- function(split) {
  train <- analysis(split)
  test <- assessment(split)
  synthesizer <- ctgan()
  train_syn <- train %>% 
    mutate(lapse_rate = lapse_count / exposure_count) %>%
    select(-lapse_count)
  synthesizer %>% 
    fit(train_syn,batch_size = 500, epochs = 300)
  syn <- synthesizer %>% 
    ctgan_sample(n = nrow(train), batch_size = 10000) %>% 
    mutate(exposure_count = exposure_count %>% 
             as.integer() %>% 
             pmax(1),
           lapse_rate = lapse_rate %>% pmin(1) %>% pmax(0),
           lapse_count = as.integer(exposure_count * lapse_rate )
    ) %>% 
    select(-lapse_rate)
  
  glm_syn <- create_glm(syn)
  glm_train <- create_glm(train)
  
  preds_syn <- predict(glm_syn, test, type = "response")
  preds_train <- predict(glm_train, test, type = "response")
  actual_response <- test$lapse_count
  
  rmse_syn <- compute_rmse(preds_syn, actual_response)
  rmse_train <- compute_rmse(preds_train, actual_response)
  
  list(rmse_syn = rmse_syn, rmse_train = rmse_train)
}

folds <- rsample::vfold_cv(modeling_data, v = 10)
result <- map(folds$splits, analyze_synthesis)
rmses <- result %>% transpose() %>% map(flatten_dbl)
rmses

# saveRDS(rmses, "shock_lapse.rds")