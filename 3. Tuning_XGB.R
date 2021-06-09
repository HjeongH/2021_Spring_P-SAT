##################################
##### 유기동물 파라미터 튜닝 #####
#####     XGBOOST, 강아지    #####
##################################

# ver1. One Hot Encoding + 5 Fold CV
# ver2. Cat Boost Encoding + 5 Fold CV (코드생략)
# ver3. Label/Onehot Encoding + 5 Fold CV (코드생략)

library(tidyverse)
library(plyr)
library(magrittr)
library(data.table)
library(gridExtra)
library(dummies)
library(xgboost)
library(caret)
library(MLmetrics)
library(progress)


# 1. Onehot Encd +cv_fold ================================
## Dataframe for Grid search ----------
set.seed(613)
tune_xgb = data.frame(
  max_depth = sample(seq(3,8),20,replace = TRUE),
  min_child_weight = sample(seq(3,8),20,replace = TRUE),
  subsample = runif(20,0.6,1),
  colsample_bytree = runif(20,0.6,1),
  eta = runif(20,0.01,0.3),
  nrounds = sample(c(500,600,700,800),20,replace = TRUE),
  acc = rep(NA,20),
  f1score = rep(NA,20)
)
tune_xgb


## Preprocess ----------
train_dog <- fread("data/final_train_dog.csv",header = TRUE,data.table = FALSE)

### categorical :: adoptionYN, neuterYN, sex, group_akc, color) 134711
### num :: weight_kg, positives, negatives, grdp, economy, hospital_num
train_dog$neuterYN = as.factor(train_dog$neuterYN)
train_dog$sex = as.factor(train_dog$sex)
train_dog$group_akc = as.factor(train_dog$group_akc)
train_dog$color = as.factor(train_dog$color)


library('fastDummies')
train_dog2 <- dummy_cols(train_dog, 
                         select_columns = c('neuterYN', 'sex', 'group_akc', 'color'),
                         remove_selected_columns = TRUE)



# 1 -------------------------------------------------------
## tuning ----------
set.seed(613)
cv = createFolds(train_dog2$adoptionYN, k = 5)

pb <- progress_bar$new(total = nrow(tune_xgb)*5) 
for(i in 1:nrow(tune_xgb)){ 
  acc_result = NULL
  f1_result = NULL
  print(paste0(i,'번째 / ',nrow(tune_xgb)))
  
  for (j in 1:5){
    index = cv[[j]]
    train = train_dog2[ -index, ]
    val = train_dog2[index, ]
    
    #scaling
    train[,c(2, 5, 6, 8, 9, 10)] <- scale(train[,c(2, 5, 6, 8, 9, 10)])
    val[,c(2, 5, 6, 8, 9, 10)] <- scale(val[,c(2, 5, 6, 8, 9, 10)])
    
    dtrain <- xgb.DMatrix(data = as.matrix(train[,-1]), label=train$adoptionYN)
    dval <- xgb.DMatrix(data = as.matrix(val[,-1]), label=val$adoptionYN)
    watchlist <- list(train=dtrain, test=dval)
    
    #xgboost modeling
    set.seed(613)
    ml_xgb <- xgb.train(data=dtrain, booster = "gblinear",  eval.metric = "logloss", objective = "binary:logistic", 
                        eta =tune_xgb$eta[i] , early_stopping_rounds = 100, 
                        max_depth=tune_xgb$max_depth[i],
                        min_child_weight = tune_xgb$min_child_weight[i],
                        subsample=tune_xgb$subsample[i],
                        colsample_bytree=tune_xgb$colsample_bytree[i], 
                        nrounds = tune_xgb$nrounds[i], watchlist = watchlist, verdose=0)
    xgb.pred = predict(ml_xgb, as.matrix(val[,-1]) )
    pre_xgb <- rep(0,nrow(val))
    pre_xgb[xgb.pred>0.5]=1
    
    acc = Accuracy(pre_xgb,  val$adoptionYN) #acc
    acc_result = c(acc_result, acc)
    f1 = F1_Score(y_pred = pre_xgb, y_true = val$adoptionYN) ##F1_score계산
    f1_result = c(f1_result, f1)
    pb$tick()
  }
  
  tune_xgb[i,'acc']=mean(acc_result)
  tune_xgb[i,'f1score']=mean(f1_result)
}


##best 6996942 0.7951265
tune_xgb = tune_xgb %>% arrange(desc(f1score))
tune_xgb %>% head(5)
write.csv(tune_xgb,"xgb_dog_dumR.csv")
#4,5,0.7836237,0.9558572,0.01042062,600,



# 2 -------------------------------------------------------
## Max_depth, Min_child tuning

## load the best ----------
best_param_sm <- tune_xgb %>% arrange(desc(f1score)) %>% head(1)
best_param_sm


## Datafame For Gird Search ----------
max_depth = seq(best_param_sm[1,]$max_depth-1,best_param_sm[1,]$max_depth+3,1)
min_child_weight = seq(best_param_sm[1,]$min_child_weight-1,best_param_sm[1,]$min_child_weight+1,1)

grid_1 = data.frame(
  max_depth = rep(max_depth,length(min_child_weight)),
  min_child_weight = rep(min_child_weight,each = length(max_depth)),
  acc = rep(NA,length(min_child_weight) * length(max_depth)),
  f1score = rep(NA,length(min_child_weight) * length(max_depth))
)

best_param_sm
grid_1$subsample <- best_param_sm$subsample
grid_1$colsample_bytree <-best_param_sm$colsample_bytree 
grid_1$eta <- best_param_sm$eta
grid_1$nrounds <- best_param_sm$nrounds
grid_1


## Tuning ----------
for(i in 1:nrow(grid_1)){ 
  acc_result = NULL
  f1_result = NULL
  print(paste0(i,'번째 / ',nrow(grid_1)))
  
  for (j in 1:5){
    index = cv[[j]]
    train = train_dog2[ -index, ]
    val = train_dog2[index, ]
    #scaling
    train[,c(2, 5, 6, 8, 9, 10)] <- scale(train[,c(2, 5, 6, 8, 9, 10)])
    val[,c(2, 5, 6, 8, 9, 10)] <- scale(val[,c(2, 5, 6, 8, 9, 10)])
    
    dtrain <- xgb.DMatrix(data = as.matrix(train[,-1]), label=train$adoptionYN)
    dval <- xgb.DMatrix(data = as.matrix(val[,-1]), label=val$adoptionYN)
    watchlist <- list(train=dtrain, test=dval)
    
    #xgboost modeling
    set.seed(613)
    ml_xgb <- xgb.train(data=dtrain, booster = "gblinear",  eval.metric = "logloss", objective = "binary:logistic", 
                        eta =grid_1$eta[i] , early_stopping_rounds = 100, 
                        max_depth=grid_1$max_depth[i],
                        min_child_weight = grid_1$min_child_weight[i],
                        subsample=grid_1$subsample[i],
                        colsample_bytree=grid_1$colsample_bytree[i], 
                        nrounds = grid_1$nrounds[i], watchlist = watchlist, verdose=0)
    
    xgb.pred = predict(ml_xgb, as.matrix(val[,-1]) )
    pre_xgb <- rep(0,nrow(val))
    pre_xgb[xgb.pred>0.5]=1
    
    acc = Accuracy(pre_xgb,  val$adoptionYN) #acc
    acc_result = c(acc_result, acc)
    f1 = F1_Score(y_pred = pre_xgb, y_true = val$adoptionYN) ##F1_score계산
    f1_result = c(f1_result, f1)
  }
  
  grid_1[i,'acc']=mean(acc_result)
  grid_1[i,'f1score']=mean(f1_result)
}


## result
grid_1 <- grid_1 %>% arrange(desc(f1score))
grid_1 %>% arrange(desc(f1score)) %>% head(1)
best_param_sm #변화없음



# 3 -------------------------------------------------------
## subsample_colsample

## Dataframe for Grid Search
subsample = seq(best_param_sm[1,]$subsample-0.1,best_param_sm[1,]$subsample+0.1,0.04)
colsample_bytree = seq(best_param_sm[1,]$colsample_bytree-0.1,best_param_sm[1,]$colsample_bytree+0.1,0.04)

grid_2 = data.frame(
  subsample = rep(subsample,length(colsample_bytree)),
  colsample_bytree = rep(colsample_bytree,each = length(subsample)),
  acc = rep(NA,length(colsample_bytree) * length(subsample)),
  f1score = rep(NA,length(colsample_bytree) * length(subsample))
)

grid_2$max_depth = grid_1$max_depth[1]
grid_2$min_child_weight = grid_1$min_child_weight[1]
grid_2$eta <- best_param_sm$eta
grid_2$nrounds <- best_param_sm$nrounds
grid_2



## Tuning 
for(i in 1:nrow(grid_2)){ 
  acc_result = NULL
  f1_result = NULL
  print(paste0(i,'번째 / ',nrow(grid_1)))
  
  for (j in 1:5){
    index = cv[[j]]
    train = train_dog2[ -index, ]
    val = train_dog2[index, ]
    #scaling
    train[,c(2, 5, 6, 8, 9, 10)] <- scale(train[,c(2, 5, 6, 8, 9, 10)])
    val[,c(2, 5, 6, 8, 9, 10)] <- scale(val[,c(2, 5, 6, 8, 9, 10)])
    
    dtrain <- xgb.DMatrix(data = as.matrix(train[,-1]), label=train$adoptionYN)
    dval <- xgb.DMatrix(data = as.matrix(val[,-1]), label=val$adoptionYN)
    watchlist <- list(train=dtrain, test=dval)
    
    #xgboost modeling
    set.seed(613)
    ml_xgb <- xgb.train(data=dtrain, booster = "gblinear",  eval.metric = "logloss", objective = "binary:logistic", 
                        eta =grid_2$eta[i] , early_stopping_rounds = 100, 
                        max_depth=grid_2$max_depth[i],
                        min_child_weight = grid_2$min_child_weight[i],
                        subsample=grid_2$subsample[i],
                        colsample_bytree=grid_2$colsample_bytree[i], 
                        nrounds = grid_2$nrounds[i], watchlist = watchlist, verdose=0)
    
    xgb.pred = predict(ml_xgb, as.matrix(val[,-1]) )
    pre_xgb <- rep(0,nrow(val))
    pre_xgb[xgb.pred>0.5]=1
    
    acc = Accuracy(pre_xgb,  val$adoptionYN) #acc
    acc_result = c(acc_result, acc)
    f1 = F1_Score(y_pred = pre_xgb, y_true = val$adoptionYN) ##F1_score계산
    f1_result = c(f1_result, f1)
  }
  
  grid_2[i,'acc']=mean(acc_result)
  grid_2[i,'f1score']=mean(f1_result)
}


## result
grid_2 <- grid_2 %>% arrange(desc(f1score))
grid_2[1,]
best_param_sm #변화 없음



# 4 -------------------------------------------------------
## eta, nrounds

## Dataframe For Grid Search
eta = seq(best_param_sm[1,]$eta-0.05,best_param_sm[1,]$eta+0.05,0.02)
nrounds = seq(best_param_sm[1,]$nrounds-100,best_param_sm[1,]$nrounds+100,100)

grid_3 = data.frame(
  eta = rep(eta,length(nrounds)),
  nrounds = rep(nrounds,each = length(eta)),
  acc = rep(NA,length(nrounds) * length(eta)),
  f1score =  rep(NA,length(nrounds) * length(eta))
)

grid_3$max_depth = grid_1$max_depth[1]
grid_3$min_child_weight = grid_1$min_child_weight[1]
grid_3$subsample <- grid_2$subsample[1]
grid_3$colsample <- grid_2$colsample[1]
grid_3<-grid_3 %>% filter(eta>0)
grid_3


## Tuning
for(i in 1:nrow(grid_3)){ 
  acc_result = NULL
  f1_result = NULL
  
  for (j in 1:5){
    index = cv[[j]]
    train = train_dog2[ -index, ]
    val = train_dog2[index, ]
    #scaling
    train[,c(2, 5, 6, 8, 9, 10)] <- scale(train[,c(2, 5, 6, 8, 9, 10)])
    val[,c(2, 5, 6, 8, 9, 10)] <- scale(val[,c(2, 5, 6, 8, 9, 10)])
    
    dtrain <- xgb.DMatrix(data = as.matrix(train[,-1]), label=train$adoptionYN)
    dval <- xgb.DMatrix(data = as.matrix(val[,-1]), label=val$adoptionYN)
    watchlist <- list(train=dtrain, test=dval)
    
    #xgboost modeling
    set.seed(613)
    ml_xgb <- xgb.train(data=dtrain, booster = "gblinear",  eval.metric = "logloss", objective = "binary:logistic", 
                        eta =grid_3$eta[i] , early_stopping_rounds = 100, 
                        max_depth=grid_3$max_depth[i], 
                        min_child_weight = grid_3$min_child_weight[i],
                        subsample=grid_3$subsample[i],
                        colsample_bytree=grid_3$colsample_bytree[i], 
                        nrounds = grid_3$nrounds[i], watchlist = watchlist, verdose=0)
    
    xgb.pred = predict(ml_xgb, as.matrix(val[,-1]) )
    pre_xgb <- rep(0,nrow(val))
    pre_xgb[xgb.pred>0.5]=1
    
    acc = Accuracy(pre_xgb,  val$adoptionYN) #acc
    acc_result = c(acc_result, acc)
    f1 = F1_Score(y_pred = pre_xgb, y_true = val$adoptionYN) ##F1_score계산
    f1_result = c(f1_result, f1)
  }
  
  grid_3[i,'acc']=mean(acc_result)
  grid_3[i,'f1score']=mean(f1_result)
}


## result
grid_3 <- grid_3 %>% arrange(desc(f1score))
grid_3[1,]


# Best combination -------------------------------------------------------
a3 = grid_3 %>% head(2)
colnames(a3)[8]<- "colsample_bytree"

a2 = grid_2 %>% head(2)
a1 = grid_1 %>% head(2)
a0 = tune_xgb %>% arrange(desc(f1score)) %>% head(2)
a3

result = rbind(a0, a1)
result = rbind(result, a2)
result = rbind(result, a3)
result %>% arrange(desc(f1score))
result <- result %>% arrange(desc(f1score))
result

write_csv(result, "xgb_dog_dum_aft.csv")


# Final ----------------------------------------------------------
# Test set -------------------------------------------------------

rm(list=ls())
result <- fread("xgb_dog_dum_aft.csv",header = TRUE,data.table = FALSE)
result = arrange(result,desc(f1score))


train_dog <- fread("data/final_train_dog.csv",header = TRUE,data.table = FALSE)
test_dog <- fread("data/final_test_dog.csv",header = TRUE,data.table = FALSE)

#preprocess
train_dog$neuterYN = as.factor(train_dog$neuterYN)
train_dog$sex = as.factor(train_dog$sex)
train_dog$group_akc = as.factor(train_dog$group_akc)
train_dog$color = as.factor(train_dog$color)

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



#scaling
train_dog2[,c(2, 5, 6, 8, 9, 10)] <- scale(train_dog2[,c(2, 5, 6, 8, 9, 10)])
test_dog2[,c(2, 5, 6, 8, 9, 10)] <- scale(test_dog2[,c(2, 5, 6, 8, 9, 10)])


dtrain <- xgb.DMatrix(data = as.matrix(train_dog2[,-1]), label=train_dog2$adoptionYN)
set.seed(613)
ml_xgb <- xgb.train(data=dtrain, booster = "gblinear",  eval.metric = "logloss", objective = "binary:logistic", 
                    eta =result$eta[1] , 
                    max_depth=result$max_depth[1],
                    min_child_weight = result$min_child_weight[1],
                    subsample=result$subsample[1],
                    colsample_bytree=result$colsample_bytree[1],
                    early_stoppind_rounds = 100,
                    watchlist = list(train=dtrain),
                    nrounds = 1500, verdose=0)


xgb.pred = predict(ml_xgb, as.matrix(test_dog2[,-1]))
pre_xgb <- rep(0,nrow(test_dog2))
pre_xgb[xgb.pred>0.5]=1


Accuracy(pre_xgb,  test_dog2$adoptionYN) #0.6969292
F1_Score(y_pred = pre_xgb, y_true = test_dog2$adoptionYN) ##0.7937067


# 2. Catboost Encd +cv_fold================================

# 3. Label/Onehot Encd +cv_fold================================


#### catboost인코딩은 파이썬에서 전처리 후 (CV별) cSV파일로 저장하여 사용함
