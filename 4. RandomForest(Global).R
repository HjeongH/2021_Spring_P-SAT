###################################
#####유기동물(강아지) 결과해석#####
#####  Random Forest, Global  #####
###################################

# Feature Importance, PDP

##
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
library(vip)        # ML global interpretation
library(pdp)        # ML global interpretation



##데이터 불러오기
train_dog <- fread("cbs/dog_train.csv",header = TRUE,data.table = FALSE)
test_dog <- fread("cbs/dog_test.csv",header = TRUE,data.table = FALSE)
train_dog$adoptionYN = as.factor(train_dog$adoptionYN)
test_dog$adoptionYN = as.factor(test_dog$adoptionYN)


# feature importance ========================================
## 파라미터 튜닝 결과 csv 이용
result <- fread("rf_dog_cbs.csv",header = TRUE,data.table = FALSE)
result = arrange(result,desc(f1score))
set.seed(613)
rf_info = randomForest(adoptionYN~., train_dog, mtry = result[1,'mtry'], ntree = result[1, 'ntree'], importance = TRUE)


## 튜닝결과 직접 입력할때는
set.seed(613)
rf_info = randomForest(adoptionYN~., train_dog, mtry = 3, ntree = 300, importance = TRUE)


## Mean Decrease Accuracy + 시각화
imp_df <- data.frame(importance(rf_info))
imp_df <- imp_df %>% 
  mutate(names = rownames(imp_df)) %>% 
  arrange(desc(MeanDecreaseAccuracy)) 

imp_df %>% 
  top_n(10, MeanDecreaseAccuracy) %>% 
  ggplot(aes(x = reorder(names, MeanDecreaseAccuracy),y = MeanDecreaseAccuracy)) +
  geom_col(fill="Orange", alpha=0.6) +
  coord_flip() +
  labs(title = "Variable Importance",
       x= "",
       y= "Mean Decrease in Accuracy") +
  theme(plot.caption = element_text(face = "italic"))+
  theme_classic()

## Mean Decrease Gini도 가능



# Partial Dependence Plot ========================================
library(pdp)
library(vip)

set.seed(613)
rf_mod_dog = randomForest(adoptionYN~., train_dog, mtry = 3, ntree = 300)

## neuterYN만 예시로 확인
partialPlot(rf_mod_dog, pred.data = train_dog, x.var = "neuterYN")