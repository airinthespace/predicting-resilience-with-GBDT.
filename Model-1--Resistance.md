---
title: "Model 1:Resistance"
author: '30910'
date: "2024-07-30"
output: 
  html_document:
    keep_md: true
---



```r
# loading libraries
library(xgboost)
library(caret)
library(dplyr)
library(MLmetrics)
library(pdp)
library(patchwork)
library(ggplot2)
library(pROC)
```


```r
# loading data
selected_df_is_resistant <- read.csv("/Users/sumeyyeoguz/Desktop/selected_df_is_resistant.csv")
```


```r
# excluding first column
selected_df_is_resistant <- selected_df_is_resistant[,-1]
```


```r
str(selected_df_is_resistant)
```

```
## 'data.frame':	140 obs. of  9 variables:
##  $ Status  : chr  "Resistant" "Non-resistant" "Non-resistant" "Non-resistant" ...
##  $ B_access: num  87.7 91.7 91.8 87.3 85.5 ...
##  $ B_use   : num  78.4 87.8 88.3 83.8 83.6 ...
##  $ B_order : num  54.4 63.6 62.5 58 60.8 ...
##  $ B_hrst  : num  35 35.1 41.2 31.9 33.9 34.6 33.7 34.4 31.9 39 ...
##  $ G_access: num  -0.5929 -0.0436 1.2196 0.0802 0.4909 ...
##  $ G_use   : num  5.96 -2.34 1.93 -2.69 -2.1 ...
##  $ G_order : num  17.54 4.53 8.46 5.38 4.44 ...
##  $ G_hrst  : num  6.286 4.843 0.971 8.15 0.295 ...
```

```r
# see how many Resistant regions 
table(selected_df_is_resistant$Status)
```

```
## 
## Non-resistant     Resistant 
##           118            22
```



```r
# Convert Status to binary (Resistant = 1, Non-resistant = 0)
selected_df_is_resistant$Status <- ifelse(selected_df_is_resistant$Status == "Resistant", 1, 0)

# Set seed for reproducibility
set.seed(123)

# Shuffle the dataset
shuffled_df <- selected_df_is_resistant[sample(nrow(selected_df_is_resistant)), ]

# Split the data into training and testing sets
trainIndex <- createDataPartition(shuffled_df$Status, p = 0.8, list = FALSE, times = 1)

train_data <- shuffled_df[trainIndex, ]
test_data <- shuffled_df[-trainIndex, ]

# Separate target and features for training data
train_target <- train_data$Status
train_features <- train_data %>% select(-Status)

# Separate target and features for test data
test_target <- test_data$Status
test_features <- test_data %>% select(-Status)

# Convert the data to matrix format for xgboost
trainMatrix <- xgb.DMatrix(data = as.matrix(train_features), label = train_target)
testMatrix <- xgb.DMatrix(data = as.matrix(test_features), label = test_target)
```

HYPERPARAMETER TUNING 


```r
# Refine the hyperparameter grid
tune_grid <- expand.grid(
  nrounds = seq(50, 1000, by = 50),   
  max_depth = seq(3, 7, by = 1),       
  eta = seq(0.01, 0.2, by = 0.1),  
  gamma = seq(0, 0.3, by = 0.1),      
  colsample_bytree = seq(0.7, 1, by = 0.1), 
  min_child_weight = seq(1, 3, by = 1), 
  subsample = seq(0.7, 1, by = 0.1) 
)
```


```r
# Set up the train control with cross-validation
train_control <- trainControl(
  method = "cv",           # Cross-validation
  number = 5,              # Number of folds
  verboseIter = TRUE,      # Print training log
  allowParallel = TRUE     # Allow parallel processing
)
```


```r
# Randomly sample a subset of 50 parameter combinations from the grid
set.seed(123)  # For reproducibility
random_subset <- tune_grid[sample(nrow(tune_grid), 50),]
```


```r
# Train the model using grid search with the refined or random subset
xgb_tuned_model <- train(
  x = train_features,
  y = as.factor(train_target),  # Make sure target is a factor
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = random_subset,     # Use the random subset of the grid
  metric = "Accuracy"           # Optimizing for accuracy
)
```




```r
# View the best parameters
best_params <- xgb_tuned_model$bestTune
print(best_params)
```

```
##    nrounds max_depth  eta gamma colsample_bytree min_child_weight subsample
## 24     250         4 0.11   0.1              0.8                1         1
```


```r
# Make predictions on the test set
predictions <- predict(xgb_tuned_model, test_features)

# Ensure you are using the correct function from caret
confusion_matrix <- caret::confusionMatrix(as.factor(predictions), as.factor(test_target))
print(confusion_matrix)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  0  1
##          0 23  4
##          1  1  0
##                                           
##                Accuracy : 0.8214          
##                  95% CI : (0.6311, 0.9394)
##     No Information Rate : 0.8571          
##     P-Value [Acc > NIR] : 0.7980          
##                                           
##                   Kappa : -0.0606         
##                                           
##  Mcnemar's Test P-Value : 0.3711          
##                                           
##             Sensitivity : 0.9583          
##             Specificity : 0.0000          
##          Pos Pred Value : 0.8519          
##          Neg Pred Value : 0.0000          
##              Prevalence : 0.8571          
##          Detection Rate : 0.8214          
##    Detection Prevalence : 0.9643          
##       Balanced Accuracy : 0.4792          
##                                           
##        'Positive' Class : 0               
## 
```

```r
final_model <- xgb_tuned_model$finalModel
```

FEATURE IMPORTANCE


```r
# Get feature importance
importance_matrix <- xgb.importance(model = final_model)

# Print feature importance
print(importance_matrix)
```

```
##     Feature       Gain      Cover  Frequency
##      <char>      <num>      <num>      <num>
## 1:    G_use 0.26942698 0.24438671 0.19721116
## 2: G_access 0.17534429 0.10432975 0.14541833
## 3:   G_hrst 0.14607942 0.16722646 0.15139442
## 4:  G_order 0.12195450 0.13007331 0.12151394
## 5: B_access 0.11275947 0.12167956 0.12749004
## 6:   B_hrst 0.10376909 0.10962464 0.10159363
## 7:    B_use 0.05525168 0.09454207 0.11553785
## 8:  B_order 0.01541458 0.02813749 0.03984064
```


```r
# Calculate total gain
total_gain <- sum(importance_matrix$Gain)
```



```r
# Calculate relative importance 
importance_matrix$Relative <- importance_matrix$Gain / sum(total_gain) * 100
print(importance_matrix)
```

```
##     Feature       Gain      Cover  Frequency  Relative
##      <char>      <num>      <num>      <num>     <num>
## 1:    G_use 0.26942698 0.24438671 0.19721116 26.942698
## 2: G_access 0.17534429 0.10432975 0.14541833 17.534429
## 3:   G_hrst 0.14607942 0.16722646 0.15139442 14.607942
## 4:  G_order 0.12195450 0.13007331 0.12151394 12.195450
## 5: B_access 0.11275947 0.12167956 0.12749004 11.275947
## 6:   B_hrst 0.10376909 0.10962464 0.10159363 10.376909
## 7:    B_use 0.05525168 0.09454207 0.11553785  5.525168
## 8:  B_order 0.01541458 0.02813749 0.03984064  1.541458
```

FEATURE IMPORTANCE PLOT 



```r
# Define growth-related features
growth_features <- c("G_access", "G_use", "G_order", "G_hrst")

# Add a classification for features
importance_matrix2 <- importance_matrix %>%
  mutate(Category = if_else(Feature %in% growth_features, "Digital Growth Variables", "Digital Endowment Variables"))


# Plot with ggplot2
ggplot(importance_matrix2, aes(x = reorder(Feature, Relative), y = Relative, fill = Category)) +
  geom_bar(stat = "identity") +
  coord_flip() +  # Flip coordinates to make features on the y-axis
  scale_fill_manual(values = c("Digital Growth Variables" = "darkgreen", "Digital Endowment Variables" = "brown")) +
  labs(title = "Model 1 Resistance Phase: Feature Importance",
       x = "Features",
       y = "Relative Importance (%)",
       fill = "Category") +
  theme_minimal()
```

![](Model-1--Resistance_files/figure-html/unnamed-chunk-17-1.png)<!-- -->

```r
# Save ggplot as an image
ggsave("/Users/sumeyyeoguz/Desktop/feature_importance_plot1.png", width = 12, height = 8)
```

PDP PLOTS


```r
# View the best parameters
print(best_params)
```

```
##    nrounds max_depth  eta gamma colsample_bytree min_child_weight subsample
## 24     250         4 0.11   0.1              0.8                1         1
```


```r
# Converting best parameters to list
best_params_list <- list(
  booster = "gbtree",
  objective = "binary:logistic",  # Binary classification
  eval_metric = "logloss",
  max_depth = best_params$max_depth,
  eta = best_params$eta,
  nthread = 2,
  subsample = best_params$subsample,
  colsample_bytree = best_params$colsample_bytree
)
```


```r
# Train the model for generating PDPs
xgb_model <- xgb.train(
  params = best_params_list,
  data = trainMatrix,
  nrounds = 250,
  watchlist = list(train = trainMatrix, test = testMatrix),
  verbose = 1
)
```


```r
# List of G predictors for which to create PDPs
g_predictors <- c("G_access", "G_use", "G_order", "G_hrst")

# Create PDPs for each predictor and store them in a list
g_pdp_plots <- lapply(g_predictors, function(pred) {
  # Compute partial dependence
  pdp_data <- partial(
    xgb_model,
    pred.var = pred,
    train = as.data.frame(test_features),
    prob = TRUE  # For binary classification, we are interested in predicted probabilities
  )
  
  # Plot PDP
  plot <- ggplot(pdp_data, aes(x = get(pred), y = yhat)) +
    geom_line() +
    labs(title = paste("PDP for", pred),
         x = pred,
         y = "Marginal Effect on Resilience") +
    theme_minimal()
  
  return(plot)
})

# Combine the PDP plots into a single visualization
g_combined_pdp_plot1 <- wrap_plots(g_pdp_plots, ncol = 2) +
  plot_annotation(
    title = "Resistance Model:Partial Dependence Plots for Growth Predictors",
    theme = theme(plot.title = element_text(color = "navy"))
  )
                    
print(g_combined_pdp_plot1)
```

![](Model-1--Resistance_files/figure-html/unnamed-chunk-22-1.png)<!-- -->

```r
# Save the combined PDP plot for model 2 G predictors
ggsave("/Users/sumeyyeoguz/Desktop/g_combined_pdp_plot1.png", g_combined_pdp_plot1, width = 12, height = 8)
```



```r
# List of B predictors for which to create PDPs
B_predictors <- c("B_access", "B_use", "B_order", "B_hrst")

# Create PDPs for each predictor and store them in a list
B_pdp_plots <- lapply(B_predictors, function(pred) {
  # Compute partial dependence
  pdp_data <- partial(
    xgb_model,
    pred.var = pred,
    train = as.data.frame(test_features),
    prob = TRUE  # For binary classification, we are interested in predicted probabilities
  )
  
  # Plot PDP
  plot <- ggplot(pdp_data, aes(x = get(pred), y = yhat)) +
    geom_line() +
    labs(title = paste("PDP for", pred),
         x = pred,
         y = "Marginal Effect on Resilience") +
    theme_minimal()
  
  return(plot)
})

# Combine the PDP plots into a single visualization
B_combined_pdp_plot1 <- wrap_plots(B_pdp_plots, ncol = 2)+
  plot_annotation(
    title = "Resistance Model: Partial Dependence Plots for Baseline Predictors",
    theme = theme(plot.title = element_text(color = "navy"))
  )
print(B_combined_pdp_plot1)
```

![](Model-1--Resistance_files/figure-html/unnamed-chunk-24-1.png)<!-- -->


```r
# Save the combined PDP plot for model 2 B predictors 
ggsave("/Users/sumeyyeoguz/Desktop/B_combined_pdp_plot1.png", B_combined_pdp_plot1, width = 12, height = 8)
```



```r
# Calculate ROC curve
# Probabilities for ROC computation
# Make predictions on the test set

predictions_prob <- predict(xgb_tuned_model, newdata = test_features, type = "prob")[,2]
roc_result <- roc(test_target, predictions_prob)
```

```
## Setting levels: control = 0, case = 1
```

```
## Setting direction: controls < cases
```

```r
# Plot ROC curve
plot(roc_result, main = "ROC Curve", col = "blue", lwd = 2)
```

![](Model-1--Resistance_files/figure-html/unnamed-chunk-26-1.png)<!-- -->

```r
# Calculate and print AUC
auc_value <- pROC::auc(roc_result)
print(paste("AUC:", auc_value))
```

```
## [1] "AUC: 0.802083333333333"
```

