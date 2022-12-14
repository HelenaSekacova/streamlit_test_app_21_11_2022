import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC,SVR
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor

import numpy as np
np.random.seed(42)


# in which columns are NaN (NOT IN THIS CASE: amount in particular columns with its importence should decide if we will keep data of columns or delete it)
def missing_values_allocation(X):
    cols={col:X[col].isnull().sum() for col in X.columns if X[col].isnull().sum()==True}
    return cols

def imputer_func(X, imputer, X_train, X_valid):

    # after choosing strategy or fill-form of imputation we train and test our dataset
    impl_X_train=pd.DataFrame(imputer.fit_transform(X_train))
    impl_X_valid=pd.DataFrame(imputer.transform(X_valid))

    # Imputation removed column names; put them back
    impl_X_train.columns=X.columns
    impl_X_valid.columns=X.columns

    # here you can check what happened with NaN (CAN BE DELETED IF WISHED)
    check_miss_train=(impl_X_train.isnull().sum().sum())
    check_miss_valid=(impl_X_valid.isnull().sum().sum())
    st.write('Check of missing values in imputed_train_columns:', check_miss_train)
    st.write(pd.DataFrame(impl_X_train, columns=X.columns))
    st.write('Check of missing values in imputed_valid_columns:', check_miss_valid)
    st.write(pd.DataFrame(impl_X_valid, columns=X.columns))

    return impl_X_train, impl_X_valid

# Scaler
def scalling(scaler,impl_X_train, impl_X_valid):
    scaled_X_train=scaler.fit_transform(impl_X_train)
    scaled_X_valid=scaler.transform(impl_X_valid)

    st.write('Scalling X_train')
    st.write(pd.DataFrame(scaled_X_train))
    st.write('Scalling X_valid')
    st.write(pd.DataFrame(scaled_X_valid))

    return scaled_X_train,scaled_X_valid

# Regression models with metrics MAE, MSE, R2
def regr_modelling(regr_model, scaled_X_train, scaled_X_valid, y_train,y_valid):
    regr_model.fit(scaled_X_train, y_train)

    y_train_pred = regr_model.predict(scaled_X_train)
    y_valid_pred = regr_model.predict(scaled_X_valid)

    MAE_train=mean_absolute_error(y_train,y_train_pred)
    MAE_valid=mean_absolute_error(y_valid,y_valid_pred)

    MSE_train=mean_squared_error(y_train,y_train_pred)
    MSE_valid=mean_squared_error(y_valid,y_valid_pred)

    R2_train=r2_score(y_train,y_train_pred)
    R2_valid=r2_score(y_valid,y_valid_pred)

    st.write('Training Data Valid Data')
    st.write('MAE', MAE_train, MAE_valid)
    st.write('MSE', MSE_train,  MSE_valid)
    st.write('R2', R2_train,  R2_valid)

# Classification model with classification_report (precision, recall, f1), confusion matrix and accuracy score
def class_modelling(class_model, scaled_X_train, scaled_X_valid, y_train,y_valid,y):
    class_model.fit(scaled_X_train, y_train)

    y_train_pred = class_model.predict(scaled_X_train)
    y_valid_pred = class_model.predict(scaled_X_valid)
    
    # Training data => plotly ff heatmap with xaxis in bottom and yaxis reversed
    st.write('Report and confusion matrix for training data')
    st.write(classification_report(y_train,y_train_pred))
    st.write(confusion_matrix(y_train,y_train_pred))
    z1=confusion_matrix(y_train,y_train_pred)
    x=y.unique().tolist()
    y=y.unique().tolist()
    z_text1=[[str(y) for y in x] for x in z1]
    fig=ff.create_annotated_heatmap(z=z1, x=x,y=y,annotation_text=z_text1,colorscale='agsunset')
    fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',xaxis = dict(title='Predicted value'),yaxis = dict(title='Real value'))
    fig['layout']['xaxis']['side'] = 'bottom'
    fig.update_layout(yaxis=dict(autorange='reversed'))
    st.write(fig)

    st.write("Accuracy score for training data:", accuracy_score(y_train, y_train_pred))

    # Valid/Testing data => plotly ff heatmap with yaxis reversed
    st.write('Report and confusion matrix for test/valid data')
    st.write(classification_report(y_valid,y_valid_pred))
    st.write(confusion_matrix(y_valid,y_valid_pred))
    z2=confusion_matrix(y_valid,y_valid_pred)
    z_text2=[[str(y) for y in x] for x in z2]
    fig=ff.create_annotated_heatmap(z=z2, x=x,y=y,annotation_text=z_text2,colorscale='tealgrn')
    fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',xaxis = dict(title='Predicted value'),yaxis = dict(title='Real value'))
    fig.update_layout(yaxis=dict(autorange='reversed'))
    st.write(fig)

    st.write("Accuracy score for testing/valid data:", accuracy_score(y_valid, y_valid_pred))

def app():
    # user input of title for page/document/app
    title=st.text_input(label='Choose title of your app', max_chars=100)
    st.title(title)
    st.caption('(predictions)')

    # selection of dataset/file
    data_file_path_train = st.file_uploader("Download train data") #Data file
    
    if data_file_path_train is None:
        st.warning("No data file uploaded")
        return

    # read data if user uploads a file
    X = pd.read_csv(data_file_path_train)
    st.write(pd.DataFrame(X))
    # seek back to position 0 after reading
    data_file_path_train.seek(0)
    st.write(X.shape)

    # drop the column if not wished in dataset; IN THIS CASE CATEGORICAL VALUES (NOT NUMERICAL)
    option= st.multiselect("Choose columns which should be processed", list(X.columns), default=list(X.columns))
    X=X[option]
        
    # plotly scatter-matrix
    dimensions = st.multiselect("Scatter matrix dimensions", list(X.columns), default=list(X.columns))
    color = st.selectbox("Color", X.columns)
    opacity = st.number_input('Choose opacity', min_value=0.1, max_value=1.0, value=1.0)

    st.write(px.scatter_matrix(X, dimensions=dimensions, color=color, opacity=opacity))

    # separating X and y from dataset
    y_column = st.selectbox("Choose a column as your target=y. Please see the bottom of the page.", X.columns)
    X.dropna(axis=0,subset=[f'{y_column}'],inplace=True)
    y=X[f'{y_column}']
    #st.write(pd.DataFrame(y))

    st.write('Has your dataset convenient distribution for separation into train and test/valid data?')
    bar_df=X[f'{y_column}'].value_counts()
    
    st.write(px.bar(X, x=X[f'{y_column}'].unique().tolist(),y=bar_df,title='Distribution of 'f"{y_column}",color=bar_df,height=400))
  
    # here you can see if any of two columns have a similarity (PCA?)
    col_selection=st.selectbox('Choose an interesting column beside your target', options=[col for col in X.columns if col!=f'{y_column}'])
    if col_selection in X.columns:
        st.write(px.box(X, x=f'{y_column}', color=f'{y_column}', y=f'{col_selection}', points="suspectedoutliers", notched=False))
    else:
        st.warning('Choose again')
    
    # Processing of missing values
    missing_val_count_by_column=(X.isnull().sum().sum())
    st.write('Number of missing values in columns:', missing_val_count_by_column)
    if missing_val_count_by_column>0:
        st.write('Location of missing values:')
        st.write(missing_values_allocation(X))

    X.drop([f'{y_column}'],axis=1,inplace=True)
    st.write(pd.DataFrame(X))

    try:
        X_train, X_valid, y_train, y_valid=train_test_split(X,y,train_size=0.8, test_size=0.2,stratify=y)
    except ValueError:
        st.markdown('''<style>.big-font{font-size:30px; color: red;}</style>''', unsafe_allow_html=True)
        st.markdown('<p class="big-font">Change your target => the least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2!', unsafe_allow_html=True)
    else:

        option=st.selectbox('Choose a replacement instead of NaN',options=['No imputer','replace_zero','replace_mean','replace_most_frequent','replace_constant'])
        if option=='No imputer':
            st.warning('Choose imputer if needed')
            imputer=None
            
        elif option=='replace_zero':
            imputer=SimpleImputer(strategy='constant',fill_value=0)

        elif option=='replace_mean':
            imputer=SimpleImputer(strategy='mean')

        elif option=='replace_most_frequent':
            imputer=SimpleImputer(strategy='most_frequent')

        elif option=='replace_constant':
            imputer=SimpleImputer(strategy='constant',fill_value=st.number_input('Fill the constant in', min_value=1, max_value=10000))
            
        # instead of else: # imputer should be empty string in this case that means there is no NaN!
        if imputer is not None:    
            impl_X_train, impl_X_valid = imputer_func(X, imputer, X_train, X_valid)
        else:
            impl_X_train, impl_X_valid=X_train, X_valid
            st.write('No imputer')

        # Scaler
        option=st.selectbox('Choose a scaler if needed', options=['No scaler','StandardScaler'])
        if option=='No scaler':
            st.warning('Choose a scaler if needed')
            scaler= None
            
        elif option=='StandardScaler':
            scaler=StandardScaler()

        if scaler is not None:
            scaled_X_train, scaled_X_valid=scalling(scaler,impl_X_train, impl_X_valid)
        else:
            st.write('No scaler')
            scaled_X_train, scaled_X_valid=impl_X_train, impl_X_valid
        
        # Model
        '''if you are using RandomForestRegressor for classification tasks, the "probability" you get might greater than 1, or less than 0. That's not what we expected.
        On the other hand, RandomForestClassifier uses accuracy_score as loss function, while RandomForestRegressor uses r2_score. That's a big difference.'''
        
        # Regression
        option=st.selectbox('Choose a regression model if avalaible',options=['No regression','LinearRegression','Lasso','SVR','RandomForestRegressor','XGBRegressor'])
        if option=='No regression':
            st.warning('Choose a classification model if avalaible')
            regr_model=None
        
        elif option=='LinearRegression':
            regr_model=LinearRegression()

        elif option=='Lasso':
            regr_model=Lasso(alpha=st.number_input('Fill the alpha (the degree of sparsity of the estimated coefficients) in', min_value=0, max_value=10000))

        elif option=='SVR':
            regr_model=SVR()

        elif option== 'RandomForestRegressor':
            regr_model=RandomForestRegressor(
            n_estimators=st.number_input('Fill the n_estimators in',min_value=1, max_value=10000),
            random_state=st.number_input('Fill the random_state in',min_value=0, max_value=10000),
            max_depth=st.number_input('Fill the max_depth in',min_value=1, max_value=10000)
            )

        elif option=='XGBRegressor':
            regr_model=XGBRegressor(
            learning_rate=st.number_input('Fill the learning_rate in',min_value=0, max_value=10000),
            n_estimators=st.number_input('Fill the n_estimators in',min_value=1, max_value=10000),
            n_jobs=st.number_input('Fill the n_jobs in',min_value=0, max_value=10000),
            random_state=st.number_input('Fill the random_state in',min_value=0, max_value=10000)
            )

        if regr_model is not None:
            regr_modelling(regr_model, scaled_X_train, scaled_X_valid, y_train,y_valid)
        else:
            st.write('It is not regression or model is not available')
        
        # Classification
        option=st.selectbox('Choose a classification model if avalaible',options=['Classification','DecisionTreeClassifier','KNeighborsClassifier','RandomForestClassifier','SVC','LogisticRegression'])
        if option=='Classification':
            st.warning('Choose one of the models')
            class_model=None
        
        elif option=='DecisionTreeClassifier':
            class_model=DecisionTreeClassifier(
            random_state=st.number_input('Fill the random_state in',min_value=0, max_value=10000),
            max_leaf_nodes=st.number_input('Fill the max_leaf_nodes in',min_value=0, max_value=10000)
            )
            
        elif option=='KNeighborsClassifier':
            class_model=KNeighborsClassifier(n_neighbors=st.number_input('Fill the n_neighbors in',min_value=0, max_value=10000))
                    
        elif option=='RandomForestClassifier':
            class_model=RandomForestClassifier(
            n_estimators=st.number_input('Fill the n_estimators in',min_value=1, max_value=10000),
            random_state=st.number_input('Fill the random_state in',min_value=0, max_value=10000),
            max_depth=st.number_input('Fill the max_depth in',min_value=1, max_value=10000)
            )
            
        elif option=='SVC':
            class_model= SVC()
            
        elif option=='LogisticRegression':
            class_model=LogisticRegression(random_state=st.number_input('Fill the random_state in',min_value=0, max_value=10000)) 
            
        if class_model is not None:
            class_modelling(class_model, scaled_X_train, scaled_X_valid, y_train,y_valid,y)
        else:
            st.write('It is not classification or model is not available')

if __name__ == "__main__":
    app()


