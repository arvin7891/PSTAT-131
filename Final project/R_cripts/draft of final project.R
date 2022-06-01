setwd('C:/Users/wilson/Desktop')
getwd()
#library(readr)
#library(ggplot2)
library(tidyverse) 
library(tidymodels)
library(corrplot) 
library(caret)
library(smotefamily) 
library(janitor) 
library(skimr)
library(patchwork)
library(lubridate)
library(ranger)

library(rlang)
library(dplyr)
library(ggplot2)
library(corrr)
library(klaR)
library(MASS)
library(discrim)
library(poissonreg)
tidymodels_prefer()

#install.packages("installr")
#install.packages("stringr")    ###
library(stringr)
library(installr)
#install.Rtools()

df <- read_csv("creditcard.csv")
#View(df)
######Data Cleaning######
head(df)
str(df)
summary(df)
# checking missing values
colSums(is.na(df))  #None of the variables have missing values
df %>% 
  clean_names()
######Data Split######
#The data was split in a 80% training, 20% testing split. 
#Stratified sampling was used as the Class distribution was skewed. (See more on that in the EDA).
#The data split was conducted prior to the EDA as I did not want to know anything about my testing data set before I tested my model on those observations.
set.seed(123)
#stratified split according to class
df_split <-df %>% 
           initial_split(df,prop = 0.8, strata = "Class")
df_train <- training(df_split)
df_test <- testing(df_split)
dim(df_train) #The training data set has about 220k+ observations 
dim(df_test)  #the testing data set has just under 50k+ observations.
df_train %>% count(Class)
df_test %>% count(Class)
######Exploratory Data Analysis######
# checking class imbalance
df %>%
  count(Class) %>% 
  mutate(prop = n/sum(n))

# class imbalance in percentage
df_train %>%
  count(Class) %>% 
  mutate(prop = n/sum(n))

common_theme <- theme(plot.title = element_text(hjust = 0.5, face = "bold"))

df_train %>%
ggplot(aes(x = factor(Class), 
                            y = stat(count), fill = factor(Class),
                            label = scales::comma(stat(count)))) +
  geom_bar(position = "dodge") + 
  geom_text(stat = 'count',
            position = position_dodge(.9), 
            vjust = -0.5, 
            size = 3) + 
  scale_x_discrete(labels = c("No Fraud", "Fraud")) +
  scale_y_continuous(labels = scales::comma)+
  labs(x = 'Class', y = 'Count') +
  ggtitle("Distribution of class labels") +
  common_theme


# Clearly the dataset is very imbalanced with 99.8% of cases being non-fraudulent transactions. 
# A simple measure like accuracy is not appropriate here as even a classifier which labels all transactions as non-fraudulent will have over 99% accuracy.
# An appropriate measure of model performance here would be AUC (Area Under the Precision-Recall Curve)
df_train %>%
  ggplot(aes(x = Time, fill = factor(Class))) + 
  geom_histogram(bins = 100)+
  labs(x = 'Time in seconds since first transaction', y = 'No. of transactions') +
  ggtitle('Distribution of time of transaction by class') +
  facet_grid(Class ~ ., scales = 'free_y') + 
  common_theme
#The ??Time?? feature looks pretty similar across both types of transactions. 
#One could argue that fraudulent transactions are more uniformly distributed, while normal transactions have a cyclical distribution
df_train %>%
ggplot(aes(x = factor(Class), y = Amount)) + 
  geom_boxplot() + 
  labs(x = 'Class', y = 'Amount') +
  ggtitle("Distribution of transaction amount by class") +
  common_theme
#There is clearly a lot more variability in the transaction values for non-fraudulent transactions.

correlations <- cor(df_train[,-1],method="pearson") %>%
corrplot( number.cex = .9, method = "circle", type = "full",tl.col = "black",title ='correlation matrix.')

####Data Prepare  
#View(df_train)
train <- df_train %>% 
  select(-Time) %>% 
  mutate(Class=factor(Class)) 
levels(train$Class)<-c("Not_Fraud", "Fraud")
train[,-30] <- scale(train[,-30])
#View(train)

test <- df_test %>% 
  select(-Time) %>% 
  mutate(Class=factor(Class))
levels(test$Class) <- c("Not_Fraud", "Fraud")
test[,-30] <- scale(test[,-30])
#View(test)
##############Choosing sampling technique##############
# downsampling
set.seed(123)
down_train <- downSample(x = train[, -ncol(train)],
                         y = train$Class)
down_train %>% 
  count(Class)%>%
  mutate(prop = n/sum(n))
# upsampling
set.seed(123)
up_train <- upSample(x = train[, -ncol(train)],
                     y = train$Class)
up_train %>% 
  count(Class)%>%
  mutate(prop = n/sum(n))
# smote
set.seed(123)
smote_train <- SMOTE(train[, -ncol(train)],train$Class)$data
colnames(smote_train)[30] <- "Class"
#View(smote_train)
smote_train%>% 
  count(Class)%>%
  mutate(prop = n/sum(n))

###compare different sampling technique using  logistic_reg model  smote is best(accuracy recall) 
#Create a recipe
recipe<-recipe(Class ~ ., data = train)
recipe
#logistic_reg  model 
log_model <- logistic_reg() %>% 
  set_engine("LiblineaR")%>% 
  set_mode("classification")
  
log_workflow <- workflow() %>% 
  add_model(log_model) %>% 
  add_recipe(recipe)

#fit the logistic model to the original training set:
set.seed(123)
ori_log_fit <- fit(log_workflow, train)
ori_log_fit %>% 
  extract_fit_parsnip() %>% 
  tidy()
#test using original test set
test_pred <- predict(ori_log_fit, new_data = test %>% select(-Class))
#metric set
test_res <- bind_cols(test_pred, test%>% select(Class))
#head(test_res)
T<-table(test_res)
TP<-T[4]
FN<-T[3]
FP<-T[2]
TN<-T[1]
accuracy=(TP+TN)/sum(T) #0.99921
recall=TP/(TP+FN) #0.61
#test_metrics<-metric_set(roc_auc)
#test_metrics(test_res, truth = Class , estimate = .pred_class)

#fit the logistic model to the down training set:
set.seed(123)
down_log_fit <- fit(log_workflow, down_train)
#test
down_test_pred <- predict(down_log_fit, new_data = test %>% select(-Class))
#metric set
down_test_res <- bind_cols(down_test_pred, test%>% select(Class))
down_T<-table(down_test_res)
TP1<-down_T[4]
FN1<-down_T[3]
FP1<-down_T[2]
TN1<-down_T[1]
down_accuracy=(TP1+TN1)/sum(down_T) #0.9721218
down_recall=TP1/(TP1+FN1) #0.89

#fit the logistic model to the up training set:
set.seed(123)
up_log_fit <- fit(log_workflow, up_train)
#test
up_test_pred <- predict(up_log_fit, new_data = test %>% select(-Class))
#metric set
up_test_res <- bind_cols(up_test_pred, test%>% select(Class))
up_T<-table(up_test_res)
TP2<-up_T[4]
FN2<-up_T[3]
FP2<-up_T[2]
TN2<-up_T[1]
up_accuracy=(TP2+TN2)/sum(up_T) #0.9759489
up_recall=TP2/(TP2+FN2) #0.9

#fit the logistic model to the smote training set   we can see smote is best
set.seed(123)
smote_log_fit <- fit(log_workflow, smote_train)
#test
smote_test_pred <- predict(smote_log_fit, new_data = test %>% select(-Class))
#metric set
smote_test_res <- bind_cols(smote_test_pred, test%>% select(Class))
smote_T<-table(smote_test_res)
TP3<-smote_T[3]
FN3<-smote_T[4]
FP3<-smote_T[1]
TN3<-smote_T[2]
smote_accuracy=(TP3+TN3)/sum(smote_T) #0.9764931
smote_recall=TP3/(TP3+FN3) #0.91


######model select using smote data#########

###logistic_reg model acc=0.9764053 recall=0.91
set.seed(123)
smote_train_folds <- vfold_cv(smote_train, v = 10) #repeat need long time
recipe<-recipe(Class ~ ., data = smote_train)
recipe

log_model <- logistic_reg(penalty = tune(), mixture = 1) %>% 
  set_engine("LiblineaR")%>% 
  set_mode('classification')

log_workflow <- workflow() %>% 
  add_model(log_model) %>% 
  add_recipe(recipe)

log_grid <- tibble(penalty = 10^seq(-3, -2, length.out = 20))

#take 1 hour
log_res <- log_workflow %>% 
  tune_grid(resamples = smote_train_folds,
            grid = log_grid,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc))

save(log_res, log_workflow, file = "D:/log_res.rda")
load("D:/log_res.rda")

autoplot(log_res, metric = "roc_auc")  ##0.002-0.003 best
show_best(log_res, metric = "roc_auc") %>% select(-.estimator, -.config)
#Final Model Building
log_workflow_tuned <- log_workflow %>% 
  finalize_workflow(select_best(log_res, metric = "roc_auc"))
log_final <- fit(log_workflow_tuned, smote_train)
#memory.limit(15000)
#Analysis of The Test Set  index:accuracy recall
log_test_res <- predict(log_final, new_data = test) %>%
  bind_cols(test %>% select(Class))
logT<-table(log_test_res)
TP4<-logT[3]
FN4<-logT[4]
FP4<-logT[1]
TN4<-logT[2]
log_accuracy=(TP4+TN4)/sum(logT) #0.9764053
log_recall=TP4/(TP4+FN4) #0.91
#log_metric <- metric_set(roc_auc)
#log_test_res %>% 
#  log_metric(truth = Class, estimate = .pred_class)

###### Boosted Tree Model 
bt_model <- boost_tree(mode = "regression",
                       min_n = tune(),
                       mtry = tune()) %>%
  set_engine("xgboost")%>%
  set_mode('classification')

bt_workflow <- workflow() %>% 
  add_model(bt_model) %>% 
  add_recipe(recipe)

bt_params <- extract_parameter_set_dials(bt_model) %>% 
  update(mtry = mtry(range= c(2, 20)),
         min_n = min_n(range= c(2, 20))
         )

# define grid
bt_grid <- grid_regular(bt_params, levels = 2)


bt_res <- bt_workflow %>% 
  tune_grid(resamples = smote_train_folds,
            grid = bt_grid,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc))

save(bt_res, bt_workflow, file = "D:/bt_res.rda")
load("D:/bt_res.rda")

autoplot(bt_res, metric = "roc_auc")  ##
show_best(bt_res, metric = "roc_auc") %>% select(-.estimator, -.config)
#Final Model Building
bt_workflow_tuned <- bt_workflow %>% 
  finalize_workflow(select_best(bt_res, metric = "roc_auc"))
bt_final <- fit(bt_workflow_tuned, smote_train)

bt_test_res <- predict(bt_final, new_data = test) %>%
  bind_cols(test %>% select(Class))
btT<-table(bt_test_res)
TP5<-btT[3]
FN5<-btT[4]
FP5<-btT[1]
TN5<-btT[2]
bt_accuracy=(TP5+TN5)/sum(btT) #0.9907833
bt_recall=TP5/(TP5+FN5) #0.86

###### SVM
svm_model <- svm_rbf( cost  = tune()
#                      ,margin = tune()
                      ) %>%
  set_engine("kernlab")%>%
  set_mode('classification')%>% 
  translate()

svm_workflow <- workflow() %>% 
  add_model(svm_model) %>% 
  add_recipe(recipe)

#svm_params <- extract_parameter_set_dials(svm_model) %>% 
#  update(cost = mtry(range= c(2, 20)),
#         min_n = min_n(range= c(2, 20))
#  )

svm_grid <- tibble(cost = 10^seq(-2, 0, length.out = 20))


svm_res <- svm_workflow %>%
  tune_grid(resamples = smote_train_folds,
            grid = svm_grid,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc))

save(svm_res, svm_workflow, file = "D:/svm_res.rda")
load("D:/svm_res.rda")

autoplot(svm_res, metric = "roc_auc")  
show_best(svm_res, metric = "roc_auc") %>% select(-.estimator, -.config)
#Final Model Building
svm_workflow_tuned <- svm_workflow %>% 
  finalize_workflow(select_best(svm_res, metric = "roc_auc"))
svm_final <- fit(svm_workflow_tuned, smote_train)
#Analysis of The Test Set  index:accuracy recall
svm_test_res <- predict(svm_final, new_data = test) %>%
  bind_cols(test %>% select(Class))
svmT<-table(svm_test_res)
TP6<-svmT[3]
FN6<-svmT[4]
FP6<-svmT[1]
TN6<-svmT[2]
svm_accuracy=(TP6+TN6)/sum(svmT) 
svm_recall=TP6/(TP6+FN6) 


###### KNN model
knn_model <- nearest_neighbor(mode = "regression",
                             neighbors = tune()
                             ) %>%
  set_engine("kknn")%>%
  set_mode('classification')

knn_workflow <- workflow() %>% 
  add_model(knn_model) %>% 
  add_recipe(recipe)

knn_params <- extract_parameter_set_dials(knn_model)

# define grid
knn_grid <- grid_regular(knn_params, levels = 4)


knn_res <- knn_workflow %>% 
  tune_grid(resamples = smote_train_folds,
            grid = knn_grid
#            ,control = control_grid(save_pred = TRUE),
#            metrics = metric_set(roc_auc)
            )

save(knn_res, knn_workflow, file = "C:/Users/DELL/Desktop/knn_res.rda")
load("C:/Users/DELL/Desktop/knn_res.rda")

autoplot(knn_res, metric = "roc_auc")  ##
show_best(knn_res, metric = "roc_auc") %>% select(-.estimator, -.config)
#Final Model Building
knn_workflow_tuned <- knn_workflow %>% 
  finalize_workflow(select_best(knn_res, metric = "roc_auc"))
knn_final <- fit(knn_workflow_tuned, smote_train)

knn_test_res <- predict(knn_final, new_data = test) %>%
  bind_cols(test %>% select(Class))
knnT<-table(knn_test_res)
TP6<-knnT[3]
FN6<-knnT[4]
FP6<-knnT[1]
TN6<-knnT[2]
knn_accuracy=(TP6+TN6)/sum(knnT) 
knn_recall=TP6/(TP6+FN6) 

#######random forest model
## using random sampling to reduce data,a tenth of the smote_train 
index<-sample(c(1:dim(smote_train)[1]),size=round(dim(smote_train)[1]/10))
sampe_smote_train<-smote_train[index,]
#View(sampe_smote_train)
#str(smote_train)
sample_smote_train_folds <- vfold_cv(sampe_smote_train, v = 10) #repeat need long time

rf_model <- rand_forest(mode = "regression",
                       min_n = tune(),
                       mtry = tune()) %>%
  set_engine("ranger")%>%
  set_mode('classification')

rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(recipe)

rf_params <- extract_parameter_set_dials(rf_model) %>% 
  update(mtry = mtry(range= c(2, 20))
#         ,min_n = min_n(range= c(2, 20))
  )

# define grid
rf_grid <- grid_regular(rf_params, levels = 2)

rf_res <- rf_workflow %>% 
  tune_grid(resamples = sample_smote_train_folds,
            grid = rf_grid,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc))

save(rf_res, rf_workflow, file = "D:/rf_res.rda")
load("D:/rf_res.rda")

autoplot(rf_res, metric = "roc_auc")  ##
show_best(rf_res, metric = "roc_auc") %>% select(-.estimator, -.config)
#Final Model Building
rf_workflow_tuned <- rf_workflow %>% 
  finalize_workflow(select_best(rf_res, metric = "roc_auc"))
rf_final <- fit(rf_workflow_tuned, smote_train)

rf_test_res <- predict(rf_final, new_data = test) %>%
  bind_cols(test %>% select(Class))
rfT<-table(rf_test_res)
TP7<-rfT[3]
FN7<-rfT[4]
FP7<-rfT[1]
TN7<-rfT[2]
rf_accuracy=(TP7+TN7)/sum(rfT) #
rf_recall=TP7/(TP7+FN7) #
