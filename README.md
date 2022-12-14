# streamlit_test_app_21_11_2022
streamlit_test_app_21_11_2022 is a simple streamlit app in python for machine learning practice . 


This app allows to see basic difference between Regression and Classification and some of their metrics.

After selection of ***one*** file which content should be divided into train and test part, it should be checked and replaced the empty/NaN values (imputation). Secondly should be decided on the basis of some choosen plots if the app will work with Regression or Classification and if any Scalling is needed before choose this.

For the Regression were used MAE, MSE and R2 metrics.

For the Classification were used classification report (which includes precision, recall and f1-score), confusion matrix, heatmap plot and accuracy score which is included in classification report as well.

1. Because of personal selection of data, there is a possibility to give a title to your page as well (limit 100characters).
2. Then should be choosen a column which will be a target for a prediction. This decision should make easier a scatter matrix following by a bar plot. (Playing with data only recommended)
3. The table 'Number of missing values in columns' provides information if the data content NaN values and where these should be found - to make a right decision which imputation is appropriate to 'replace' them.
4. As mentioned above Scalling should take part only in case of data with various value range. (Data with high value range will dominate calculations)
5. The next tasks depend on the data which should be processed.
- Regression 

 : models available: LinearRegression, Lasso, SVR, RandomForestRegressor or XGBRegressor
 
 - Classification
 
 : models available: DecisionTreeClassifier, KNeighborsClassifier, RandomForestClassifier, SVC, LogisticRegression

*For some of the models is there an opportunity to change their parameters*
