
library(tidyverse)
library(caret) # for using some ML models
library(DMwR) # for smote implementation
library(purrr) # for functional programming (map)
library(pROC) # for AUC calculations
library(PRROC) # for Precision-Recall curve calculations
library(xgboost)
library(ROCR)

datapath <- "~/Dropbox/ML-credit/sato_06112018/all_vals_race1_interestrate1.csv"
outputpath <- "~/Dropbox/ML-credit/code_R/ML-credit/output/"

set.seed(123)

test_roc <- function(model, data) {
  roc(data$default,
      predict(model, newdata=data, type="prob")[,1])
}

# Import Data
data_df <- read_csv(datapath)
race_vars = data_df  %>% select(starts_with("race"))
data_df = data_df %>% select(-starts_with("race"))

all_data <- data_df %>% dplyr::select(-X1, -IsTestData, -Default2, -IsCalibrateData,-cur_int_rate, 
                                   -starts_with("Logit"), -starts_with("RandomForest"), 
                                   -starts_with("_merg"),
                                   -sato) %>%
                                  mutate(Default = as.factor(Default)) %>% rename(default = Default)
train_data <- data_df %>% filter(IsTestData == "left_only" & IsCalibrateData == "left_only") %>%
  dplyr::select(-X1, -IsTestData, -Default2, -IsCalibrateData, -cur_int_rate, -sato,
                -starts_with("Logit"), -starts_with("RandomForest"), -starts_with("_merg")) %>%
  mutate(Default = as.factor(Default)) %>% rename(default = Default)
cal_data <- data_df %>% filter(IsTestData == "left_only" & IsCalibrateData == "both") %>%
  dplyr::select(-X1, -IsTestData, -Default2, -IsCalibrateData, -cur_int_rate, -sato,
                -starts_with("Logit"), -starts_with("RandomForest"), -starts_with("_merg")) %>%
  mutate(Default = as.factor(Default)) %>% rename(default = Default)
test_data <- data_df %>% filter(IsTestData == "both")  %>% 
  dplyr::select(-X1, -IsTestData, -Default2, -IsCalibrateData, -cur_int_rate, -sato,
                -starts_with("Logit"), -starts_with("RandomForest"), -starts_with("_merg")) %>%
  mutate(Default = as.factor(Default)) %>% rename(default = Default)
levels(train_data$default) <- c("current", "default")
levels(test_data$default) <- c("current", "default")


# Construct Samples
sampleIndex2 <- createDataPartition(train_data$default, p = 0.40)
sampleIndex2 <- sampleIndex2$Resample1
sample_train <- train_data[sampleIndex2, ] %>% filter(is.na(default) == FALSE)
sampleIndex2 <- createDataPartition(test_data$default, p = 0.40)
sampleIndex2 <- sampleIndex2$Resample1
sample_test <- test_data[sampleIndex2, ] %>% filter(is.na(default) == FALSE)
print("Random Subsample Test data size in RAM:");
print(object.size(sample_train), units = 'Mb')

# Prep XGBoost
sample_train$default <- as.numeric(sample_train$default) - 1
output_train <- as.matrix(as.numeric(sample_train$default))
training <- as.matrix(sample_train %>% dplyr:::select(-default))
xgb.train <- xgb.DMatrix(data = training, label = output_train, missing = NaN)

cal_data$default <- as.numeric(cal_data$default) - 1
output_cal <- as.matrix(as.numeric(cal_data$default))
calibrate <- as.matrix(cal_data %>% dplyr:::select(-default))
xgb.cal<- xgb.DMatrix(data = calibrate, label = output_cal, missing = NaN)

test_data$default  <- as.numeric(test_data$default) - 1
output_test <- as.matrix(as.numeric(test_data$default))
test <- as.matrix(test_data %>% dplyr:::select(-default))
xgb.test <- xgb.DMatrix(data = test, label = output_test, missing = NaN)

output_all <- as.matrix(as.numeric(all_data$default))
all <- as.matrix(all_data %>% dplyr:::select(-default) )
xgb.all <- xgb.DMatrix(data = all, label = output_all, missing = NaN)

# all_sato0 <- as.matrix(all_data %>% dplyr:::select(-default) %>% mutate(sato = 0) )
# xgb.all_sato0 <- xgb.DMatrix(data = all_sato0, label = output_all, missing = NaN)

# train the xgboost learner
# start <- Sys.time()
# xgbmod <- xgb.cv(data = xgb.train,
#                  max_depth = 6,
#                  nfold = 3,
#                  nrounds = 200,
#                  objective = "binary:logistic",
#                  metrics = "rmse",
#                  eval_metric = "map", eval_metric = "auc", eval_metric = "rmse",
#                  eta = .25,
#                  verbose = 1,
#                  early_stopping_rounds = 10)
# print(Sys.time() - start)

start <- Sys.time()
xgbmodel <- xgb.train(data = xgb.train,
                 max_depth = 6, 
                 nrounds = 120, 
                 objective = "binary:logistic",
                 metrics = "rmse", 
                 eta = .1,
                 verbose = 1)
print(Sys.time() - start)

# importance <- xgb.importance(feature_names = colnames(xgb.train), model = xgbmodel)
# head(importance)


pred.test <- predict(xgbmodel, xgb.test)
pred.test.df <- bind_cols(data_frame(output_test[,1]), data_frame(pred.test))
colnames(pred.test.df) <- c("outcome", "xgboost")
roc_test <- pROC:::roc( pred.test.df$outcome, pred.test.df$xgboost , algorithm = 2) 

pred.cal <- predict(xgbmodel, xgb.test)
pred.cal.df <- bind_cols(data_frame(output_test[,1]), data_frame(pred.cal))
colnames(pred.cal.df) <- c("outcome", "xgboost")
roc_test <- pROC:::roc( pred.cal.df$outcome, pred.cal.df$xgboost , algorithm = 2) 


cal_plot_data <- calibration(as.factor(outcome) ~ xgboost, data = pred.cal.df, class = 1)$data
ggplot() + xlab("Bin Midpoint") +
  geom_line(data = cal_plot_data, aes(midpoint, Percent),
            color = "#F8766D") +
  geom_point(data = cal_plot_data, aes(midpoint, Percent),
             color = "#F8766D", size = 3) +
  geom_line(aes(c(0, 100), c(0, 100)), linetype = 2, 
            color = 'grey50')

xgboost_output <- predict(xgbmodel, xgb.all)
xgb.save(xgbmodel, 'xgb.model.race.no.int')
pred.all <- bind_cols(data_df, data_frame(xgboost_output), race_vars)
write_csv(pred.all, "~/Dropbox/ML-credit/code_R/ML-credit/output/xgboost_output_w_race_noint.csv")
