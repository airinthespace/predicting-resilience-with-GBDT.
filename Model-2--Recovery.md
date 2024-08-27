---
title: "Model 2:Recovery"
author: '30910'
date: "2024-08-13"
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
library(ggplot2)
library(patchwork) 
library(pROC)
```


```r
# loading data
selected_df <- read.csv("/Users/sumeyyeoguz/Desktop/selected_df.csv")
```


```r
# exluding first column
selected_df <- selected_df[,-1]
```


```r
str(selected_df)
```

```
## 'data.frame':	140 obs. of  9 variables:
##  $ Status  : chr  "Recovered" "Recovered" "Recovered" "Recovered" ...
##  $ B_access: num  87.7 91.7 91.8 87.3 85.5 ...
##  $ B_use   : num  78.4 87.8 88.3 83.8 83.6 ...
##  $ B_order : num  54.4 63.6 62.5 58 60.8 ...
##  $ B_hrst  : num  35 35.1 41.2 31.9 33.9 34.6 33.7 34.4 31.9 39 ...
##  $ G_access: num  3.15 -0.48 4.88 4.52 8.29 ...
##  $ G_use   : num  8.44 1.95 7.01 5.6 7.09 ...
##  $ G_order : num  7.6047 4.9851 7.9789 8.1538 -0.0823 ...
##  $ G_hrst  : num  3.71 5.7 4.37 3.76 7.67 ...
```


```r
# Dropping "Resistant" status from the data
notresistant_df <- selected_df %>% filter(Status != "Resistant")
```


```r
table(selected_df$Status)
```

```
## 
## Not recovered     Recovered     Resistant 
##            18           102            20
```


```r
table(notresistant_df$Status)
```

```
## 
## Not recovered     Recovered 
##            18           102
```



```r
# Binary classification with xgboost based on Status column

# Convert Status to binary (Recovered = 1, Not recovered = 0)
notresistant_df$Status <- ifelse(notresistant_df$Status == "Recovered", 1, 0)

# Separate features and target
features <- notresistant_df %>% select(-Status)
target <- notresistant_df$Status

# Split the data into training and testing sets
set.seed(123)
train_index <- createDataPartition(target, p = 0.8, list = FALSE)
train_features <- features[train_index, ]
train_target <- target[train_index]
test_features <- features[-train_index, ]
test_target <- target[-train_index]

# Convert to matrix format for xgboost
train_matrix <- xgb.DMatrix(data = as.matrix(train_features), label = train_target)
test_matrix <- xgb.DMatrix(data = as.matrix(test_features), label = test_target)
```

HYPERPARAMETER TUNING 


```r
# Refine the hyperparameter grid
tune_grid <- expand.grid(
  nrounds = seq(50, 1000, by = 50),     
  max_depth = seq(3, 7, by = 1),        
  eta = seq(0.01, 0.2, by = 0.05),      
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
##   nrounds max_depth  eta gamma colsample_bytree min_child_weight subsample
## 7      50         4 0.01   0.1              0.7                2       0.9
```


```r
# Make predictions on the test set
predictions <- predict(xgb_tuned_model, newdata = test_features)


conf_matrix <- confusionMatrix(as.factor(predictions), as.factor(test_target))
print(conf_matrix)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  0  1
##          0  0  0
##          1  3 21
##                                           
##                Accuracy : 0.875           
##                  95% CI : (0.6764, 0.9734)
##     No Information Rate : 0.875           
##     P-Value [Acc > NIR] : 0.6476          
##                                           
##                   Kappa : 0               
##                                           
##  Mcnemar's Test P-Value : 0.2482          
##                                           
##             Sensitivity : 0.000           
##             Specificity : 1.000           
##          Pos Pred Value :   NaN           
##          Neg Pred Value : 0.875           
##              Prevalence : 0.125           
##          Detection Rate : 0.000           
##    Detection Prevalence : 0.000           
##       Balanced Accuracy : 0.500           
##                                           
##        'Positive' Class : 0               
## 
```


```r
final_model <- xgb_tuned_model$finalModel
```


```r
# feature importance
importance <- xgb.importance(model = final_model)
importance
```

```
##     Feature        Gain       Cover   Frequency
##      <char>       <num>       <num>       <num>
## 1: G_access 0.280146253 0.247869796 0.198630137
## 2:  B_order 0.277621944 0.293824524 0.308219178
## 3:    B_use 0.164252797 0.113501881 0.150684932
## 4:  G_order 0.141593592 0.147380836 0.130136986
## 5:    G_use 0.071875742 0.100951853 0.102739726
## 6:   B_hrst 0.047756860 0.068867738 0.068493151
## 7:   G_hrst 0.010662014 0.025635858 0.034246575
## 8: B_access 0.006090798 0.001967514 0.006849315
```



```r
# Calculate total gain
total_gain <- sum(importance$Gain)

# Calculate relative importance 
importance$Relative <- importance$Gain / sum(total_gain) * 100
print(importance)
```

```
##     Feature        Gain       Cover   Frequency   Relative
##      <char>       <num>       <num>       <num>      <num>
## 1: G_access 0.280146253 0.247869796 0.198630137 28.0146253
## 2:  B_order 0.277621944 0.293824524 0.308219178 27.7621944
## 3:    B_use 0.164252797 0.113501881 0.150684932 16.4252797
## 4:  G_order 0.141593592 0.147380836 0.130136986 14.1593592
## 5:    G_use 0.071875742 0.100951853 0.102739726  7.1875742
## 6:   B_hrst 0.047756860 0.068867738 0.068493151  4.7756860
## 7:   G_hrst 0.010662014 0.025635858 0.034246575  1.0662014
## 8: B_access 0.006090798 0.001967514 0.006849315  0.6090798
```

FEATURE IMPORTANCE PLOT



```r
# Define growth-related features
growth_features <- c("G_access", "G_use", "G_order", "G_hrst")

# Add a classification for features
importance_matrix2 <- importance %>%
  mutate(Category = if_else(Feature %in% growth_features, "Digital Growth Variables", "Digital Endowment Variables"))


# Plot with ggplot2
ggplot(importance_matrix2, aes(x = reorder(Feature, Relative), y = Relative, fill = Category)) +
  geom_bar(stat = "identity") +
  coord_flip() +  # Flip coordinates to make features on the y-axis
  scale_fill_manual(values = c("Digital Growth Variables" = "darkgreen", "Digital Endowment Variables" = "brown")) +
  labs(title = "Model 2 Recovery Phase: Relative Importance",
       x = "Features",
       y = "Relative Importance (%)",  # Update the y-axis label to reflect the new metric
       fill = "Category") +
  theme_minimal()
```

![](Model-2--Recovery_files/figure-html/unnamed-chunk-18-1.png)<!-- -->


```r
# Save the ggplot 
ggsave("/Users/sumeyyeoguz/Desktop/feature_importance_plot2.png", width = 12, height = 8)
```


PDP PLOTS


```r
# View the best parameters
print(best_params)
```

```
##   nrounds max_depth  eta gamma colsample_bytree min_child_weight subsample
## 7      50         4 0.01   0.1              0.7                2       0.9
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
  data = train_matrix,
  nrounds = 100,
  watchlist = list(train = train_matrix, test = test_matrix),
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
g_combined_pdp_plot2 <- wrap_plots(g_pdp_plots, ncol = 2) +
  plot_annotation(
    title = "Recovery Model:Partial Dependence Plots for Growth Predictors",
    theme = theme(plot.title = element_text(color = "navy"))
  )
                    
print(g_combined_pdp_plot2)
```

![](Model-2--Recovery_files/figure-html/unnamed-chunk-23-1.png)<!-- -->


```r
# Save the combined PDP plot for model 2 G predictors
ggsave("/Users/sumeyyeoguz/Desktop/g_combined_pdp_plot2.png", g_combined_pdp_plot2, width = 12, height = 8)
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
B_combined_pdp_plot2 <- wrap_plots(B_pdp_plots, ncol = 2)+
  plot_annotation(
    title = "Recovery Model: Partial Dependence Plots for Baseline Predictors",
    theme = theme(plot.title = element_text(color = "navy"))
  )
print(B_combined_pdp_plot2)
```

![](Model-2--Recovery_files/figure-html/unnamed-chunk-25-1.png)<!-- -->


```r
# Save the combined PDP plot for model 2 B predictors 
ggsave("/Users/sumeyyeoguz/Desktop/B_combined_pdp_plot2.png", B_combined_pdp_plot2, width = 12, height = 8)
```


```r
# Calculate ROC curve
# Probabilities for ROC computation
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

![](Model-2--Recovery_files/figure-html/unnamed-chunk-27-1.png)<!-- -->

```r
# Calculate and print AUC
auc_value <- auc(roc_result)
print(paste("AUC:", auc_value))
```

```
## [1] "AUC: 0.801587301587302"
```
