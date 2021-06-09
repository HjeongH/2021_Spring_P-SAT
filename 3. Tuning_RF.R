##################################
##### 유기동물 파라미터 튜닝 #####
#####  Random Forest, 강아지 #####
##################################

# ver1. Cat Boost Encoding + 5 Fold CV
# ver2. One Hot Encoding + 5 Fold CV
# ver3. Label/Onehot Encoding + 5 Fold CV


library(dplyr)
library(tidyverse)
library(ggplot2)
library(data.table)
library(gridExtra)
library(reshape2)
library(lubridate)
library(RColorBrewer)
library(rlang)
library(corrplot)
library(gridExtra)
library(DMwR)
library(randomForest)
library(progress)
library(tictoc)
library(MLmetrics)
library(caret)


# 1. Catboost Encd +cv_fold================================
rm(list = ls())

## Dataframe for Grid search
sqrt(11)  #3.316625 

tune_rf <- expand.grid(mtry = c(3,4,5,6,10), 
                       ntree = c(50,100,200,300))
tune_rf$acc = NA
tune_rf$f1score = NA

tune_rf

## modeling
pb <- progress_bar$new(total = nrow(tune_rf)*5) 

for(i in 1:nrow(tune_rf)){ 
  acc_result = NULL
  f1_result = NULL
  print(paste0(i,'번째'))
  
  for (j in 1:5){
    
    train <- fread(paste0("cbs//dog_train_cv",j,".csv"))
    val <- fread(paste0("cbs//dog_val_cv",j,".csv"))
    train$adoptionYN = as.factor(train$adoptionYN)
    val$adoptionYN = as.factor(val$adoptionYN)
    
    set.seed(613)
    rf_mod_1 = randomForest(adoptionYN~., train, mtry = tune_rf[i,'mtry'], ntree = tune_rf[i, 'ntree'])
    rf_pred_1 = predict(rf_mod_1, newdata = select(val, -adoptionYN))
    
    #acc, f1
    acc = Accuracy(rf_pred_1,  val$adoptionYN) #acc
    acc_result = c(acc_result, acc)
    f1 = F1_Score(y_pred = rf_pred_1, y_true = val$adoptionYN) ##F1_score계산
    f1_result = c(f1_result, f1)
    pb$tick()}
  
  tune_rf[i,'acc']=mean(acc_result)
  tune_rf[i,'f1score']=mean(f1_result)
  
}

tune_rf = tune_rf %>% arrange(desc(f1score))
write.csv(tune_rf,"rf_dog_cbs.csv")


## test 
result <- fread("rf_dog_cbs.csv",header = TRUE,data.table = FALSE)
result = arrange(result,desc(f1score))

train_dog <- fread("cbs/dog_train.csv",header = TRUE,data.table = FALSE)
test_dog <- fread("cbs/dog_test.csv",header = TRUE,data.table = FALSE)
train_dog$adoptionYN = as.factor(train_dog$adoptionYN)
test_dog$adoptionYN = as.factor(test_dog$adoptionYN)

set.seed(613)
rf_mod_1 = randomForest(adoptionYN~., train_dog, mtry = result[1,'mtry'], ntree = result[1, 'ntree'])
rf_pred_1 = predict(rf_mod_1, newdata = select(test_dog, -adoptionYN))

Accuracy(rf_pred_1,  test_dog$adoptionYN) #0.7360648
F1_Score(y_pred = rf_pred_1, y_true = test_dog$adoptionYN) ##0.8001264





# 2. Onehot Encd +cv_fold ================================
rm(list = ls())

## Dataframe for Grid search
sqrt(11)  #3.316625

tune_rf <- expand.grid(mtry = c(3,4,5,6,10), 
                       ntree = c(50,100,200,300))
tune_rf$acc = NA
tune_rf$f1score = NA

tune_rf


## Preprogress
pb <- progress_bar$new(total = nrow(tune_rf)*5) 

train_dog <- fread("data/final_train_dog.csv",header = TRUE,data.table = FALSE)

train_dog %>% colnames
train_dog$neuterYN = as.factor(train_dog$neuterYN)
train_dog$sex = as.factor(train_dog$sex)
train_dog$group_akc = as.factor(train_dog$group_akc)
train_dog$color = as.factor(train_dog$color)

train_dog %>% str

library('fastDummies')
train_dog2 <- dummy_cols(train_dog, 
                         select_columns = c('neuterYN', 'sex', 'group_akc', 'color'),
                         remove_selected_columns = TRUE)


## Modeling
set.seed(613)
cv = createFolds(train_dog2$adoptionYN, k = 5)

for(i in 1:nrow(tune_rf)){ 
  acc_result = NULL
  f1_result = NULL
  print(paste0(i,'번째 / ',nrow(tune_rf)))
  
  
  for (j in 1:5){
    index = cv[[j]]
    train = train_dog2[ -index, ]
    val = train_dog2[index, ]
    
    train$adoptionYN = as.factor(train$adoptionYN)
    val$adoptionYN = as.factor(val$adoptionYN)
    
    train[,c(2, 5, 6, 8, 9, 10)] <- scale(train[,c(2, 5, 6, 8, 9, 10)])
    val[,c(2, 5, 6, 8, 9, 10)] <- scale(val[,c(2, 5, 6, 8, 9, 10)])
    
    set.seed(613)
    rf_mod_1 = randomForest(adoptionYN~., train, mtry = tune_rf[i,'mtry'], ntree = tune_rf[i, 'ntree'])
    rf_pred_1 = predict(rf_mod_1, newdata = select(val, -adoptionYN))
    

    acc = Accuracy(rf_pred_1,  val$adoptionYN) #acc
    acc_result = c(acc_result, acc)
    f1 = F1_Score(y_pred = rf_pred_1, y_true = val$adoptionYN) ##F1_score계산
    f1_result = c(f1_result, f1)
    pb$tick()}
  
  tune_rf[i,'acc']=mean(acc_result)
  tune_rf[i,'f1score']=mean(f1_result)
  
}

tune_rf = tune_rf %>% arrange(desc(f1score))
write.csv(tune_rf,"rf_dog_dum.csv")



## test 
result <- fread("rf_dog_dum.csv",header = TRUE,data.table = FALSE)
result = arrange(result,desc(f1score))

train_dog <- fread("data/final_train_dog.csv",header = TRUE,data.table = FALSE)
test_dog <- fread("data/final_test_dog.csv",header = TRUE,data.table = FALSE)





# 3. Label/Onehot Encd +cv_fold================================
rm(list = ls())

## Preprogress
train_dog$adoptionYN = as.factor(train_dog$adoptionYN)
train_dog$neuterYN = as.factor(train_dog$neuterYN)
train_dog$sex = as.factor(train_dog$sex)
train_dog$group_akc = as.factor(train_dog$group_akc)
train_dog$color = as.factor(train_dog$color)

test_dog$adoptionYN = as.factor(test_dog$adoptionYN)
test_dog$neuterYN = as.factor(test_dog$neuterYN)
test_dog$sex = as.factor(test_dog$sex)
test_dog$group_akc = as.factor(test_dog$group_akc)
test_dog$color = as.factor(test_dog$color)

train_dog2 <- dummy_cols(train_dog, 
                         select_columns = c('neuterYN', 'sex', 'group_akc', 'color'),
                         remove_selected_columns = TRUE)

test_dog2 <- dummy_cols(test_dog, 
                        select_columns = c('neuterYN', 'sex', 'group_akc', 'color'),
                        remove_selected_columns = TRUE)


## scaling
train_dog2[,c(2, 5, 6, 8, 9, 10)] <- scale(train_dog2[,c(2, 5, 6, 8, 9, 10)])
test_dog2[,c(2, 5, 6, 8, 9, 10)] <- scale(test_dog2[,c(2, 5, 6, 8, 9, 10)])

set.seed(613)
rf_mod_1 = randomForest(adoptionYN~., train_dog, mtry = result[1,'mtry'], ntree = result[1, 'ntree'])
rf_pred_1 = predict(rf_mod_1, newdata = select(test_dog, -adoptionYN))


## test
Accuracy(rf_pred_1,  test_dog$adoptionYN) #0.7280541
F1_Score(y_pred = rf_pred_1, y_true = test_dog$adoptionYN) #0.7926317
