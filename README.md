# 2021 봄학기 P-SAT 주제분석
성균관대학교 통계분석학회 P-SAT 2021년 봄학기 선형대수학팀 주제분석

### << 유기동물 입양 예측 모델링 >>
**황정현(팀장)🙋**, 김지민, 고경현, 반경림, 전효림

- **분석목표**  :  유기동물 입양에 유의한 변수 파악 및 입양 예측 모델링
- **분석기간**  :  2021.04.19 ~ 2020.05.07 (3주)
- **분석도구**  :  R, Python
- **분석내용**  :  
  - 데이터 수집 : 크롤링, api, 국가통계포털
  - 전처리 및 시각화
  - 변수검정
      T-test, Logistic Regression Test, Chi-square Independence Test, Cramer's V
  - 입양여부 예측
    - 데이터 처리 : 스케일링(표준화), 인코딩(one-hot, label, catboost)
    - 파라미터튜닝 :  5 fold CV
    - 모델종류 :  Decision Tree, RandomForest, Logistic Regression, CatBoost, AdaBoost, LGBM, XGBoost, SVM, Naive Bayes Classification
  - 모델링 결과해석
    - Logistic Regression : Odds Ratio
    - RandomForest :  Feature Importance, Partial Dependence Plot, LIME
  - 활용 방안
       
- **본인역할**  :  팀 총괄, 특징변수(자연어) 전처리 및 시각화, 하이퍼파라미터튜닝(XGBoost, LGBM, RandomForest), LIME
