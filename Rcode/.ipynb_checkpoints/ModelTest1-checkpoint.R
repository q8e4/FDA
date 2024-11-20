# Load necessary libraries
library(refund)   # For pffr model
library(FDboost)  # For FDboost model
library(tidyverse)  # Data wrangling

# Create results folder
if (!dir.exists("results"))
  dir.create("results")

# Data import
dta <- readRDS("data/data_comb.RDS")
names(dta)

### Data Preparation for pffr and FDboost ###

# Get variables---------------------------------------------------------------
response_vars <- names(dta)[grep("grf|knee_moment|hip_moment|ankle_moment", names(dta))]
pred_vars <- names(dta)[grep("angle|vel|accl", names(dta))]

# Restructure data -----------------------------------------------------------
dta_mat <- dta[map_lgl(dta, is.matrix)]
dta_x <- dta_mat[grepl("accl|vel|angle", names(dta_mat))]
dta_y <- dta_mat[grepl("grf|knee_moment|hip_moment|ankle_moment", names(dta_mat))]

# Define training and test sets
set.seed(42)
train_ind <- sample(1:nrow(dta_y[[1]]), floor(0.8 * nrow(dta_y[[1]])))
test_ind <- setdiff(1:nrow(dta_y[[1]]), train_ind)

# Prepare response variable (using ankle moment for demonstration)
response <- "ankle_moment_ap"
y_train <- dta_y[[response]][train_ind, ]
y_test <- dta_y[[response]][test_ind, ]

# Prepare predictor variables for functional data analysis
x_train_fun <- lapply(dta_x, function(x) x[train_ind, ])
x_test_fun <- lapply(dta_x, function(x) x[test_ind, ])

# Cycle variable (ensure proper alignment)
cycle <- dta$cycle
if (length(cycle) != ncol(dta_x[[1]])) {
  stop("The number of time points in cycle does not match the functional covariates.")
}

# Prepare data for scalar covariates
scalar_vars <- c("study", "sex", "cond", "side", "step", "age", "ht", "wt")
x_sca <- model.matrix(~ -1 + ., data = as.data.frame(dta[scalar_vars]))
x_train_sca <- x_sca[train_ind, ]
x_test_sca <- x_sca[test_ind, ]

### pffr Model ###
print("Starting pffr model...")

# Create formula for pffr with explicit grid specification
form <- paste(response, " ~ 1 + ", paste(
  paste0("ff(", pred_vars, ", yind = cycle, xind = 1:", ncol(dta_x[[1]]), ")"),
  collapse = " + "),
  " + age + ht + wt + sex + s(cond, bs = 're') + s(id, bs = 're')"
)

# Convert train and test data to data frames
train_df <- as.data.frame(lapply(dta, function(x) if (!is.null(dim(x))) return(I(x[train_ind, ])) else x[train_ind]))
train_df$id <- as.factor(train_df$id)
train_df$cond <- as.factor(train_df$cond)
train_df$sex <- as.factor(train_df$sex)

test_df <- as.data.frame(lapply(dta, function(x) if (!is.null(dim(x))) return(I(x[test_ind, ])) else x[test_ind]))
test_df$id <- as.factor(test_df$id)
test_df$cond <- as.factor(test_df$cond)
test_df$sex <- as.factor(test_df$sex)

# Fit the pffr model
pffr_model <- pffr(as.formula(form),
                   yind = 1:ncol(dta_x[[1]]),  # Ensure yind matches the time points
                   algorithm = "bam",
                   discrete = TRUE,  # Enable discrete processing
                   data = train_df,
                   method = "fREML")

# Prediction with pffr model
prediction_pffr <- predict(pffr_model, newdata = test_df)

saveRDS(prediction_pffr, file = "results/prediction_pffr.RDS")

rm(pffr_model, prediction_pffr); gc()

print("pffr model completed.")

### FDboost Model ###
print("Starting FDboost model...")

# Create formula for FDboost
form_fdboost <- paste(response, " ~ 1 + ", paste(
  paste0("bsignal(", pred_vars, ", cycle)"),
  collapse = " + "),
  " + bbsc(age) + bbsc(ht) + bbsc(wt) + bolsc(sex, df = 2) + brandomc(cond)"
)

# Convert train data to list for FDboost
train_list <- as.list(dta)
train_list$cycle <- cycle
train_list[pred_vars] <- lapply(train_list[pred_vars], function(x) scale(x, scale = FALSE))
train_list <- lapply(train_list, function(x) if (!is.null(dim(x))) return(I(x[train_ind, ])) else x[train_ind])

test_list <- as.list(dta)
test_list$cycle <- cycle
test_list[pred_vars] <- lapply(test_list[pred_vars], function(x) scale(x, scale = FALSE))
test_list <- lapply(test_list, function(x) if (!is.null(dim(x))) return(I(x[test_ind, ])) else x[test_ind])

# Fit the FDboost model
fdboost_model <- FDboost(as.formula(form_fdboost),
                         data = train_list,
                         timeformula = ~ bbs(cycle, df = 5),
                         control = boost_control(mstop = 50, nu = 0.1))  # Reduced for faster debugging

# Prediction with FDboost model
prediction_fdboost <- predict(fdboost_model, newdata = test_list)

saveRDS(prediction_fdboost, file = "results/prediction_fdboost.RDS")

rm(fdboost_model, prediction_fdboost); gc()

print("FDboost model completed.")
