---
title: "Data Preprocessing"
author: '30910'
date: "2024-07-30"
output: 
  html_document:
    keep_md: true
---




```r
library(readr)
library(dplyr)
library(reshape2)
library(tidyr)
```

outcome variable: employment rates


```r
employment_rates <- read_csv("/Users/sumeyyeoguz/Desktop/EMPLOYMENT RATES/Employmentrates.csv", 
    col_types = cols(DATAFLOW = col_skip(), 
        `LAST UPDATE` = col_skip(), freq = col_skip(), 
        unit = col_skip(), sex = col_skip(), 
        age = col_skip(), OBS_FLAG = col_skip()))
employment_rates <- employment_rates %>% rename(year = TIME_PERIOD, Employment_rates = OBS_VALUE)
```

predictors: digital development level indicators


```r
internet_access <- read_csv("/Users/sumeyyeoguz/Desktop/DIGITAL/Households with access to the internet at home-19-20-21-22-23, 187/isoc_r_iacc_h_page_linear.csv", 
    col_types = cols(DATAFLOW = col_skip(), 
        `LAST UPDATE` = col_skip(), freq = col_skip(), 
        unit = col_skip(), OBS_FLAG = col_skip()))
internet_access <- internet_access %>% rename(year = TIME_PERIOD, Internet_access = OBS_VALUE)
```


```r
internet_usage <- read_csv("/Users/sumeyyeoguz/Desktop/DIGITAL/Individuals regularly using the internet 19-20-21-22-23, 187 /tgs00050_page_linear.csv", 
    col_types = cols(DATAFLOW = col_skip(), 
        `LAST UPDATE` = col_skip(), freq = col_skip(), 
        indic_is = col_skip(), unit = col_skip(), 
        OBS_FLAG = col_skip()))
internet_usage <- internet_usage %>% rename(year = TIME_PERIOD, Internet_usage = OBS_VALUE)
```


```r
online_order <- read_csv("/Users/sumeyyeoguz/Desktop/DIGITAL/ ordered goods or services over the internet for private use 19-20-21-22-23/isoc_r_blt12_i_page_linear.csv", 
    col_types = cols(DATAFLOW = col_skip(), 
        `LAST UPDATE` = col_skip(), freq = col_skip(), 
        indic_is = col_skip(), unit = col_skip(), 
        OBS_FLAG = col_skip()))
online_order <- online_order %>% rename(year = TIME_PERIOD, Online_order = OBS_VALUE)
```


```r
hr_sci_tech <- read_csv("/Users/sumeyyeoguz/Desktop/DIGITAL/HR in Science and Tech/hrst_st_rcat_page_linear.csv", 
    col_types = cols(DATAFLOW = col_skip(), 
        `LAST UPDATE` = col_skip(), freq = col_skip(), 
        category = col_skip(), unit = col_skip(), 
        OBS_FLAG = col_skip()))

hr_sci_tech <- hr_sci_tech %>% rename(year = TIME_PERIOD, HR_sci_tech = OBS_VALUE)
```


```r
# List of dataframes to be merged
df_list <- list(internet_access, internet_usage, online_order, hr_sci_tech)

# Merge all dataframes on 'GEO' and 'TIME_PERIOD'
df <- Reduce(function(x, y) merge(x, y, by = c("geo", "year"), all = TRUE), df_list)
```


```r
# exluding regions with missing values from the merged_digital dataset
df <- df %>%
  filter(!is.na(Internet_access) & !is.na(Internet_usage) & !is.na(Online_order) & !is.na(HR_sci_tech))
```


```r
# drop 2023 data
df <- df %>%
  filter(year != 2023)
```


```r
# only keep regions with data for all years
df <- df %>%
  group_by(geo) %>%
  filter(n_distinct(year) == 4)
```


```r
# summary statistics for the digital development indicators
digital_vars_distribiton <- df %>%
  select(Internet_access, Internet_usage, Online_order, HR_sci_tech) %>%
  summary()
```

```
## Adding missing grouping variables: `geo`
```

```r
print(digital_vars_distribiton)
```

```
##      geo            Internet_access  Internet_usage   Online_order  
##  Length:560         Min.   : 66.15   Min.   :60.37   Min.   :14.49  
##  Class :character   1st Qu.: 87.18   1st Qu.:81.64   1st Qu.:50.52  
##  Mode  :character   Median : 91.69   Median :87.50   Median :63.36  
##                     Mean   : 90.84   Mean   :86.29   Mean   :62.15  
##                     3rd Qu.: 95.04   3rd Qu.:93.02   3rd Qu.:76.04  
##                     Max.   :100.00   Max.   :99.23   Max.   :93.32  
##   HR_sci_tech   
##  Min.   :13.30  
##  1st Qu.:25.18  
##  Median :31.65  
##  Mean   :32.45  
##  3rd Qu.:38.33  
##  Max.   :57.30
```

```r
# Calculate the percentage difference for employment rates between 2019 and 2020
emp_19_20 <- employment_rates %>%
  filter(year %in% c(2019, 2020)) %>%
  spread(year, Employment_rates) %>%
  mutate(Emp_19_20 = ((`2020` - `2019`) / `2019`) * 100) %>%
  select(geo, Emp_19_20)

# Calculate the percentage difference for employment rates between 2019 and 2021
emp_19_21 <- employment_rates %>%
  filter(year %in% c(2019, 2021)) %>%
  spread(year, Employment_rates) %>%
  mutate(Emp_19_21 = ((`2021` - `2019`) / `2019`) * 100) %>%
  select(geo, Emp_19_21)


emp_19_22 <- employment_rates %>%
  filter(year %in% c(2019, 2022)) %>%
  spread(year, Employment_rates) %>%
  mutate(Emp_19_22 = ((`2022` - `2019`) / `2019`) * 100) %>%
  select(geo, Emp_19_22)
```


```r
# Merge the percentage difference data frames  with merge()
df <- merge(df, emp_19_20, by = "geo")
df <- merge(df, emp_19_21, by = "geo")
df <- merge(df, emp_19_22, by = "geo")
```


```r
str(df)
```

```
## 'data.frame':	560 obs. of  9 variables:
##  $ geo            : chr  "AT11" "AT11" "AT11" "AT11" ...
##  $ year           : num  2019 2020 2021 2022 2019 ...
##  $ Internet_access: num  87.7 87.2 92.9 90.5 91.7 ...
##  $ Internet_usage : num  78.4 83.1 86 85 87.8 ...
##  $ Online_order   : num  54.4 64 58.5 58.6 63.6 ...
##  $ HR_sci_tech    : num  35 37.2 36.2 36.3 35.1 36.8 38.2 37.1 41.2 41.6 ...
##  $ Emp_19_20      : num  0 0 0 0 -1.02 ...
##  $ Emp_19_21      : num  -0.663 -0.663 -0.663 -0.663 -1.148 ...
##  $ Emp_19_22      : num  1.459 1.459 1.459 1.459 0.128 ...
```


```r
# Calculate growth rates for the digital development indicators 

# Calculate the percentage difference for Internet_access between 2019 and 2020
internet_access_19_20 <- internet_access %>%
  filter(year %in% c(2019, 2020)) %>%
  spread(year, Internet_access) %>%
  mutate(Internet_access_19_20 = ((`2020` - `2019`) / `2019`) * 100) %>%
  select(geo, Internet_access_19_20)

# Calculate the percentage difference for Internet_access between 2019 and 2021
internet_access_19_21 <- internet_access %>%
  filter(year %in% c(2019, 2021)) %>%
  spread(year, Internet_access) %>%
  mutate(Internet_access_19_21 = ((`2021` - `2019`) / `2019`) * 100) %>%
  select(geo, Internet_access_19_21)

# Calculate the percentage difference for Internet_access between 2019 and 2022
internet_access_19_22 <- internet_access %>%
  filter(year %in% c(2019, 2022)) %>%
  spread(year, Internet_access) %>%
  mutate(Internet_access_19_22 = ((`2022` - `2019`) / `2019`) * 100) %>%
  select(geo, Internet_access_19_22)
```


```r
# Merge the percentage difference data frames with merge()
df <- merge(df, internet_access_19_20, by = "geo")
df <- merge(df, internet_access_19_21, by = "geo")
df <- merge(df, internet_access_19_22, by = "geo")
```


```r
# Calculate the percentage difference for Internet_usage between 2019 and 2020
internet_usage_19_20 <- internet_usage %>%
  filter(year %in% c(2019, 2020)) %>%
  spread(year, Internet_usage) %>%
  mutate(Internet_usage_19_20 = ((`2020` - `2019`) / `2019`) * 100) %>%
  select(geo, Internet_usage_19_20)

# Calculate the percentage difference for Internet_usage between 2019 and 2021
internet_usage_19_21 <- internet_usage %>%
  filter(year %in% c(2019, 2021)) %>%
  spread(year, Internet_usage) %>%
  mutate(Internet_usage_19_21 = ((`2021` - `2019`) / `2019`) * 100) %>%
  select(geo, Internet_usage_19_21)

# Calculate the percentage difference for Internet_usage between 2019 and 2022
internet_usage_19_22 <- internet_usage %>%
  filter(year %in% c(2019, 2022)) %>%
  spread(year, Internet_usage) %>%
  mutate(Internet_usage_19_22 = ((`2022` - `2019`) / `2019`) * 100) %>%
  select(geo, Internet_usage_19_22)
```


```r
# Merge the percentage difference data frames with merge()
df <- merge(df, internet_usage_19_20, by = "geo")
df <- merge(df, internet_usage_19_21, by = "geo")
df <- merge(df, internet_usage_19_22, by = "geo")
```


```r
# Calculate the percentage difference for Online_order between 2019 and 2020
online_order_19_20 <- online_order %>%
  filter(year %in% c(2019, 2020)) %>%
  spread(year, Online_order) %>%
  mutate(Online_order_19_20 = ((`2020` - `2019`) / `2019`) * 100) %>%
  select(geo, Online_order_19_20)

# Calculate the percentage difference for Online_order between 2019 and 2021
online_order_19_21 <- online_order %>%
  filter(year %in% c(2019, 2021)) %>%
  spread(year, Online_order) %>%
  mutate(Online_order_19_21 = ((`2021` - `2019`) / `2019`) * 100) %>%
  select(geo, Online_order_19_21)

# Calculate the percentage difference for Online_order between 2019 and 2022
online_order_19_22 <- online_order %>%
  filter(year %in% c(2019, 2022)) %>%
  spread(year, Online_order) %>%
  mutate(Online_order_19_22 = ((`2022` - `2019`) / `2019`) * 100) %>%
  select(geo, Online_order_19_22)
```


```r
# Merge the percentage difference data frames with merge()
df <- merge(df, online_order_19_20, by = "geo")
df <- merge(df, online_order_19_21, by = "geo")
df <- merge(df, online_order_19_22, by = "geo")
```


```r
# Calculate the percentage difference for HR_sci_tech between 2019 and 2020
hr_sci_tech_19_20 <- hr_sci_tech %>%
  filter(year %in% c(2019, 2020)) %>%
  spread(year, HR_sci_tech) %>%
  mutate(HR_sci_tech_19_20 = ((`2020` - `2019`) / `2019`) * 100) %>%
  select(geo, HR_sci_tech_19_20)

# Calculate the percentage difference for HR_sci_tech between 2019 and 2021
hr_sci_tech_19_21 <- hr_sci_tech %>%
  filter(year %in% c(2019, 2021)) %>%
  spread(year, HR_sci_tech) %>%
  mutate(HR_sci_tech_19_21 = ((`2021` - `2019`) / `2019`) * 100) %>%
  select(geo, HR_sci_tech_19_21)

# Calculate the percentage difference for HR_sci_tech between 2019 and 2022
hr_sci_tech_19_22 <- hr_sci_tech %>%
  filter(year %in% c(2019, 2022)) %>%
  spread(year, HR_sci_tech) %>%
  mutate(HR_sci_tech_19_22 = ((`2022` - `2019`) / `2019`) * 100) %>%
  select(geo, HR_sci_tech_19_22)
```


```r
# Merge the percentage difference data frames with merge()
df <- merge(df, hr_sci_tech_19_20, by = "geo")
df <- merge(df, hr_sci_tech_19_21, by = "geo")
df <- merge(df, hr_sci_tech_19_22, by = "geo")
```


```r
# Keeping only 2019 data for research purposes
df <- df %>%
  filter(year == 2019)
```


```r
# Drop the year and geo column
df <- df %>%
  select(-year, -geo)
```


```r
# Create a new column empty
df$G_Internet_access = NA
df$G_Internet_usage = NA
df$G_Online_order = NA
df$G_HR_sci_tech = NA
df$Status = NA
```


```r
# Define the function to update the data frame based on Emp_19_22 values
update_status_and_growth <- function(df, i) {
  if (df$Emp_19_22[i] > 0) {
      df$Status[i] = "Recovered"
      df$G_Internet_access[i] = df$Internet_access_19_22[i]
      df$G_Internet_usage[i] = df$Internet_usage_19_22[i]
      df$G_Online_order[i] = df$Online_order_19_22[i]
      df$G_HR_sci_tech[i] = df$HR_sci_tech_19_22[i]
    } else {
      df$Status[i] = "Not recovered"
      df$G_Internet_access[i] = df$Internet_access_19_22[i]
      df$G_Internet_usage[i] = df$Internet_usage_19_22[i]
      df$G_Online_order[i] = df$Online_order_19_22[i]
      df$G_HR_sci_tech[i] = df$HR_sci_tech_19_22[i]
    }

  return(df)
}
```


```r
df_is_resistant <- df
```


```r
# Define function to update resistance status and growth rates
update_resistance_status <- function(df, i, status) {
  df$Status[i] = status
  df$G_Internet_access[i] = df$Internet_access_19_20[i]
  df$G_Internet_usage[i] = df$Internet_usage_19_20[i]
  df$G_Online_order[i] = df$Online_order_19_20[i]
  df$G_HR_sci_tech[i] = df$HR_sci_tech_19_20[i]
  return(df)
}
```


```r
for (i in 1:nrow(df)) {
  if (df$Emp_19_20[i] > 0) {
    df <- update_resistance_status(df, i, "Resistant")
    df_is_resistant <- update_resistance_status(df_is_resistant, i, "Resistant")
      
  
  } else if (df$Emp_19_20[i] == 0) {
    
    df_is_resistant <- update_resistance_status(df_is_resistant, i, "Resistant")
    
    if (df$Emp_19_21[i] >= 0) {
      df$Status[i] = "Resistant"
      df$G_Internet_access[i] = df$Internet_access_19_21[i]
      df$G_Internet_usage[i] = df$Internet_usage_19_21[i]
      df$G_Online_order[i] = df$Online_order_19_21[i]
      df$G_HR_sci_tech[i] = df$HR_sci_tech_19_21[i]
      
    } else {
      df <- update_status_and_growth(df, i)
      
    }
  } else {
    df_is_resistant <- update_resistance_status(df_is_resistant, i, "Non-resistant")
    
   df <- update_status_and_growth(df, i)
  }
}
```


```r
# Drop the columns that are not needed
selected_df <- df %>%
  select(Status, Internet_access, Internet_usage, Online_order, HR_sci_tech, G_Internet_access, G_Internet_usage, G_Online_order, G_HR_sci_tech)
```


```r
# Drop the columns that are not needed
selected_df_is_resistant <- df_is_resistant %>%
  select(Status, Internet_access, Internet_usage, Online_order, HR_sci_tech, G_Internet_access, G_Internet_usage, G_Online_order, G_HR_sci_tech)
```


```r
# Rename columns of selected_df for better readability
selected_df <- selected_df %>%
  rename(
    B_access = Internet_access,
    G_access = G_Internet_access,
    B_use = Internet_usage,
    G_use = G_Internet_usage,
    B_order = Online_order,
    G_order = G_Online_order,
    B_hrst = HR_sci_tech,
    G_hrst = G_HR_sci_tech
  )
```


```r
# Rename columns of selected_df_is_resistant for better readability
selected_df_is_resistant <- selected_df_is_resistant %>%
  rename(
    B_access = Internet_access,
    G_access = G_Internet_access,
    B_use = Internet_usage,
    G_use = G_Internet_usage,
    B_order = Online_order,
    G_order = G_Online_order,
    B_hrst = HR_sci_tech,
    G_hrst = G_HR_sci_tech
  )
```


```r
# Save the selected_df to a csv file
write.csv(selected_df, "/Users/sumeyyeoguz/Desktop/selected_df.csv")
```


```r
# Save the selected_df to a csv file
write.csv(selected_df_is_resistant, "/Users/sumeyyeoguz/Desktop/selected_df_is_resistant.csv")
```

