###################################
#####유기동물(강아지) 결과해석#####
#####  Random Forest, Local   #####
###################################

# LIME

## 디렉토리설정과 패키지 불러오기
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
library(randomForest)
library(lime)       # ML local interpretation


## 데이터 불러오기+전처리
train_dog <- fread("data/final_train_dog.csv",header = TRUE,data.table = FALSE)
test_dog <- fread("data/final_test_dog.csv",header = TRUE,data.table = FALSE)

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

levels(train_dog$adoptionYN) <- c("No", "Yes")
levels(test_dog$adoptionYN) <- c("No", "Yes")

## 스케일링
train_dog[,c(2, 5, 6, 8, 9, 10)] <- scale(train_dog[,c(2, 5, 6, 8, 9, 10)])
test_dog[,c(2, 5, 6, 8, 9, 10)] <- scale(test_dog[,c(2, 5, 6, 8, 9, 10)])


## 처리속도/양 문제로 LIME으로 확인할 test, train을 재설정
set.seed(613)
idx1 <- createDataPartition(train_dog$adoptionYN, p=0.01)
train=train_dog[unlist(idx1),] #nrow 560
set.seed(613)
idx2 <- createDataPartition(test_dog$adoptionYN, p=0.001)
test = test_dog[unlist(idx2),] #nrow 24


## 모델링
set.seed(613)
fit.caret <- train(
  adoptionYN ~ ., 
  data = train, 
  method = 'ranger',
  trControl = trainControl(method = "cv", number = 5, classProbs = TRUE),
  tuneLength = 1,
  importance = 'impurity'
)


## LIME
trainx= dplyr::select(train, -adoptionYN)
testx= dplyr::select(test, -adoptionYN)

explainer <- lime(trainx, fit.caret, n_bins = 6, quantile_bins = TRUE)

explanation_df99 <- lime::explain(testx, explainer, 
                                  n_labels = 2, 
                                  n_features = 10,   dist_fun = "gower",
                                  kernel_width = sqrt(10),
                                  n_permutations = 500, 
                                  feature_select = "highest_weights")


plot_explanations(explanation_df99)

explanation_df1 <- lime::explain(testx, explainer, 
                                 labels = "Yes", 
                                 n_features = 10,   dist_fun = "gower",
                                 kernel_width = sqrt(10),
                                 n_permutations = 500, 
                                 feature_select = "highest_weights")

plot_explanations(explanation_df1)
plot_features(explanation_df1)