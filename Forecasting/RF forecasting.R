# Forecsting remainders with OLS

## Libraries
library(tidyverse)
library(forecast)
library(ggplot2)
library(dplyr)
library(data.table)
library(IRdisplay)
library(progress)

library(foreach)
library(doParallel)

library(caret)
library(randomForest)

## Custom functions
display_limited <- function(dt) {
  n <- nrow(dt)
  
  # If there are 20 or fewer rows, display the full table
  if (n <= 20) {
    limited_dt <- dt
  } else {
    # Otherwise, concatenate the first 5 rows, '...' and the last 5 rows
    limited_dt <- rbind(head(dt, 5), as.list(rep("...", ncol(dt))), tail(dt, 5))
  }
  
  # Generate raw HTML manually
  html_output <- paste0(
    "<table border='1' style='border-collapse:collapse;'>",
    "<thead><tr>",
    paste0("<th>", colnames(limited_dt), "</th>", collapse = ""),
    "</tr></thead>",
    "<tbody>",
    paste0(
      apply(limited_dt, 1, function(row) {
        paste0("<tr>", paste0("<td>", row, "</td>", collapse = ""), "</tr>")
      }),
      collapse = ""
    ),
    "</tbody></table>"
  )
  
  # Display the HTML in the Jupyter notebook
  display_html(html_output)
}                                # Display tables
calculate_metrics <- function(R_t, R_hat_t, individual) {
  # Ensure the inputs are numeric vectors and individual is a dataframe
  if (!is.numeric(R_t) || !is.numeric(R_hat_t)) {
    stop("Both R_t and R_hat_t need to be numeric vectors.")
  }
  
  # Calculate metrics
  mae <- mean(abs(R_t - R_hat_t), na.rm = TRUE)
  rmse <- sqrt(mean((R_t - R_hat_t)^2, na.rm = TRUE))
  mape <- mean(abs((R_t - R_hat_t) / R_t), na.rm = TRUE) * 100
  r_squared <- ifelse(all(R_t == R_hat_t), 1, summary(lm(R_t ~ R_hat_t))$r.squared)
  
  # Create a data frame to hold the metrics and values
  metrics_table <- data.frame(
    MAE = mae,
    RMSE = rmse,
    MAPE = mape,
    R_squared = r_squared
  )
  
  # Return the metrics table
  return(metrics_table)
}        # Calculate metrics
prepare_X_t <- function(individual) {
  # Ensure the input is a dataframe
  if (!is.data.frame(individual)) {
    stop("The input must be a dataframe.")
  }
  
  # Extract hour from start_time and create a 'time_of_day' column
  individual$time_of_day <- format(as.POSIXct(individual$HourDK), "%H:%M:%S")
  
  # Exclude specified columns but keep 'time_of_day'
  X_t <- subset(individual, select = -c(HourDK, GrossConsumptionMWh))
  
  # Convert month, weekday, and time_of_day to factors with a reference category
  X_t$month <- relevel(as.factor(X_t$MonthOfYear), ref = "December")  # Set December as reference
  X_t$weekday <- relevel(as.factor(X_t$DayOfWeek), ref = "Sunday")   # Set Sunday as reference 
  X_t$time_of_day <- relevel(as.factor(X_t$Hour), ref = "0")         # Set 23 (11 PM) as reference
  
  # Remove original 'MonthOfYear', 'DayOfWeek', and 'Hour' columns to avoid duplication
  X_t <- subset(X_t, select = -c(MonthOfYear, DayOfWeek, Hour))
  
  # Create dummy variables for all factor columns (excluding reference levels)
  X_t <- model.matrix(~ . - 1, data = X_t)
  
  # Find the column indices for numerical columns AFTER creating dummy variables
  num_cols <- grep("^(Electric cars|Plug-in hybrid cars|humidity_past1h|temp_mean_past1h|wind_speed_past1h|EL_price)", colnames(X_t))
  
  # Standardize selected numerical columns
  X_t[, num_cols] <- apply(X_t[, num_cols], 2, 
                           function(x) (x - min(x)) / (max(x) - min(x)))
  
  # Return the processed dataframe
  return(as.data.frame(X_t))
}                            # Data Preparation
lag_and_align_data <- function(X_t, R_t, h = 1) {
  # Validate inputs
  if (!is.numeric(R_t)) {
    stop("R_t should be a numeric vector.")
  }
  if (!is.data.frame(X_t) && !is.matrix(X_t)) {
    stop("X_t should be a dataframe or a matrix.")
  }
  if (!is.numeric(h) || h < 1) {
    stop("h should be a positive integer.")
  }
  
  # Convert X_t to a dataframe if it's a matrix
  if (is.matrix(X_t)) {
    X_t <- as.data.frame(X_t)
  }
  
  # Align R_t with the lagged X_t
  # Shift R_t by h positions to align with X_t from the previous timestep
  R_t_aligned <- R_t[(h + 1):length(R_t)]
  
  # Keep X_t up to the second to last row, so it aligns with the shifted R_t
  X_t_aligned <- X_t[1:(nrow(X_t) - h), ]
  
  # Return the aligned datasets
  list(X_t = X_t_aligned, R_t = R_t_aligned)
}                # Lag and Align data by \\(h\\) (horizon)
plot_actual_vs_estimated <- function(R_t, R_hat_t, individual) {
  # Validate input
  if (!is.numeric(R_t) || !is.numeric(R_hat_t)) {
    stop("R_t and R_hat_t should be numeric vectors.")
  }
  if (!is.data.frame(individual)) {
    stop("individual should be a dataframe.")
  }
  
  # Create the plot
  plot(R_t, type = 'l', col = 'blue', xlab = "Time", ylab = "Value", 
       main = "Actual vs. Estimated Time Series\nelvarme: %s, zip_code: %s")
  lines(R_hat_t, type = 'l', col = 'red')
  legend("topleft", legend = c("Actual", "Estimated"), col = c("blue", "red"), lty = 1)
} # Plot actual vs estimated

## Loading data
##### Setting workign directory and loadign data #####
base_path <- "C:/Users/madsh/OneDrive/Dokumenter/kandidat/Fællesmappe/Forecasting-energy-consumption/Data Cleaning"
setwd(base_path)
data <- fread(paste0(base_path,"/Output_file.csv"))
MSTL <- fread(paste0(base_path,"/MSTL_decomp_results.csv"))


## Parameters
#train_size    <- 17544 #2 year training set
train_size    <- 8784  #1 year training set
num_timesteps <- 720
h             <- 1
ntree         <- 100
total_size    <- nrow(data)-h

## Hyper parameter tuning
individual <- data

X_t <- prepare_X_t(as.data.frame(individual))
R_t <- as.matrix(MSTL$Remainder, nrow = 26292, ncol = 1)

lag_and_align <- lag_and_align_data(X_t, R_t)
X_t <- as.matrix(lag_and_align$X_t)
R_t <- as.numeric(as.matrix(lag_and_align$R_t))

# Hyper-parameter tuning to find the best mtry value
#tuned_results <- tuneRF(X_t, R_t, stepFactor = 1.5,
#improve = 0.01, ntreeTry = 100, trace = TRUE, plot = TRUE)
#best_mtry <- tuned_results[1, "mtry"]
best_mtry <- 36


## Data preparation
X_t           <- prepare_X_t(as.data.frame(data))
R_t           <- as.matrix(MSTL$Remainder, nrow = 26304, ncol = 1)

lag_and_align <- lag_and_align_data(X_t, R_t, h=h)
X_t           <- as.matrix(lag_and_align$X_t)
R_t           <- as.matrix(lag_and_align$R_t)


### Fitting without parallel
# Initialize the progress bar
pb <- progress_bar$new(
  format = "(:spin) [:bar] :percent [Elapsed time: :elapsedfull || Estimated time remaining: :eta]",
  total = length(seq(1, nrow(X_t) - train_size, by = num_timesteps)), 
  complete = "=", 
  incomplete = "-", 
  current = ">", 
  clear = FALSE, 
  width = 100
)

results <- list()  # Initialize an empty list to store results

no_cores <- detectCores()-12
cl <- makeCluster(no_cores)
registerDoParallel(cl)

results <- foreach(j = seq(1, nrow(X_t) - train_size, by = num_timesteps), .combine = 'c', .packages = 'randomForest') %dopar% {
  # Define the range for the training set
  start_index <- j
  end_index <- j + train_size - 1
  
  # Prepare the training set
  train_X_t <- X_t[start_index:end_index, ]
  train_R_t <- R_t[start_index:end_index]
  
  # Train the Random Forest model
  rf_model <- randomForest(x = train_X_t, y = train_R_t, ntree=ntree, mtry=best_mtry)
  
  # Prepare the test set
  test_start_index <- end_index + 1
  test_end_index <- min(end_index + num_timesteps, total_size)
  test_X_t <- X_t[test_start_index:test_end_index, , drop = FALSE]
  
  # Predict on the test set
  test_predictions <- predict(rf_model, newdata = test_X_t)
  
  # Return the predictions
  num_predictions_to_return <- min(num_timesteps, total_size - test_start_index + 1)
  return(test_predictions[1:num_predictions_to_return])
}
stopCluster(cl)

R_hat_t <- unlist(results)


### Calculating metrics
individual_metrics <- calculate_metrics(tail(R_t,n=length(R_t)-train_size), R_hat_t, data)
display_limited(individual_metrics)



## Save the results
# Define the file path
path_R <- "C:/Users/madsh/OneDrive/Dokumenter/kandidat/Fællesmappe/Forecasting-energy-consumption/Data/Results/RF/R_hat_t"
path_M <- "C:/Users/madsh/OneDrive/Dokumenter/kandidat/Fællesmappe/Forecasting-energy-consumption/Data/Results/RF/Metrics"
# Automatically generate file names based on the parameters
file_name_R_hat_t <- file.path(path_R, paste0("h=", h, "_steps_ahead=", num_timesteps, "_ntree=", ntree, "_RF_R_hat_t.csv"))
file_name_metrics <- file.path(path_M, paste0("h=", h, "_steps_ahead=", num_timesteps, "_ntree=", ntree, "_RF_Metrics.csv"))

# Save R_hat_t as a CSV file
write.csv(R_hat_t, file = file_name_R_hat_t, row.names = FALSE)

# Save individual_metrics as a CSV file
write.csv(individual_metrics, file = file_name_metrics, row.names = FALSE)












################################################################################
################################################################################
                       #non-Parallel processing
################################################################################
################################################################################
for (j in seq(1, nrow(X_t) - train_size, by = num_timesteps)) {
  # Update the progress bar
  pb$tick()
  
  # Define the range for the training set
  start_index <- j
  end_index <- j + train_size - 1
  
  # Prepare the training set
  train_X_t <- X_t[start_index:end_index, ]
  train_R_t <- R_t[start_index:end_index]
  
  # Train the Random Forest model
  rf_model <- randomForest(x = train_X_t, y = train_R_t, ntree = ntree, mtry = best_mtry)
  
  # Prepare the test set
  test_start_index <- end_index + 1
  test_end_index <- min(end_index + num_timesteps, total_size)
  test_X_t <- X_t[test_start_index:test_end_index, , drop = FALSE]
  
  # Predict on the test set
  test_predictions <- predict(rf_model, newdata = test_X_t)
  
  # Determine the number of predictions to return
  num_predictions_to_return <- min(num_timesteps, total_size - test_start_index + 1)
  
  # Append the predictions to the results list
  results[[length(results) + 1]] <- test_predictions[1:num_predictions_to_return]
}

# Combine the results into a single vector
R_hat_t <- unlist(results)
















