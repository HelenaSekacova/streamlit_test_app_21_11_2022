import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,OneHotEncoder,LabelBinarizer
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC,SVR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,roc_auc_score,make_scorer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import types
from itertools import count
from collections import Counter

import numpy as np
np.random.seed(42)

# add @cache (@cache_data) to defs
# Regression - > transformation https://scikit-learn.org/stable/modules/compose.html#transformed-target-regressor
# try to add ROC-curve PLOT and PIPELINE WITH CROSS-VALIDATION
# scaler for every column and multiselect => only add minmax
# add to models 'opt_mod=globals()[opt]()'
# cross val score + cross validate => problematic classification. both binary and multiple
# cross val score + cross validate =>  StratifiedKFold over KFold when dealing with classification tasks with imbalanced class distributions
## https://stackoverflow.com/questions/68284264/does-the-pipeline-object-in-sklearn-transform-the-test-data-when-using-the-pred
## https://www.kaggle.com/code/pouryaayria/a-complete-ml-pipeline-tutorial-acu-86#3.1.-Evaluate-Algorithms:-Baseline
## https://www.kaggle.com/code/rakesh2711/multiple-models-using-pipeline/notebook



# choice if the rows with missing values should stay in. 'Continue' leaves without changes. For deleting need to be set up the procentage how many precents of missing values are not accepted.
def deleting_missing_rows(X,percentage_rows):
    X['missing']=X.apply(lambda x:x.isnull().sum(),axis='columns')
    count_X_rows=len(X.columns)
    df_missing_rows=pd.DataFrame()
    for row in X['missing']:
        if float(row)>=float(count_X_rows*percentage_rows/100):
            miss_row=X.loc[X['missing']==float(row)]
            df_missing_rows=pd.concat([df_missing_rows,miss_row],axis=0).drop_duplicates()
    'Rows which are missing more than wished percentage of their values: '
    st.write(df_missing_rows)
    st.write(df_missing_rows.shape)
    st.markdown('<p style="color:Red;font-size:15px;font-weight: bold;">Should be deleted the chosen rows? If yes, press the \'Delete rows\' button</p>',unsafe_allow_html=True )
    if f'deleting_rows_for_encoding' not in st.session_state:
        st.session_state[f'deleting_rows_for_encoding']=['Continue','Delete']
    st.selectbox('Delete rows',st.session_state[f'deleting_rows_for_encoding'],key='deleting_rows')
    
    if st.session_state['deleting_rows']=='Delete':
        X=X[~X['missing'].isin(df_missing_rows['missing'])]
        X.drop(['missing'],axis=1,inplace=True)
        X
        X.shape
        return X 
    elif st.session_state['deleting_rows']=='Continue':
        X.drop(['missing'],axis=1,inplace=True)
        X
        X.shape
        return X        

# choice if the columns with missing values should be considered as well
def info_about_cols(X,percentage_cols):
    count_X_cols=len(X)
    dict_missing_cols={col:X[col].isnull().sum() for col in X if (X[col].isnull().sum())>(percentage_cols/100*count_X_cols)}
    st.markdown('<p style="color:Red;font-size:20px;font-weight: bold;">Boolean columns will be converted to float => True:1,False:0. No need to drop.</p>',unsafe_allow_html=True )
    list_missing_cols=[]
    for mis_numb in dict_missing_cols.keys():
        list_missing_cols.append(mis_numb)
    list_missing_cols
    st.markdown('<p style="color:Red;font-size:15px;font-weight: bold;">Should be deleted the chosen columns? If yes, press the \'Delete columns\' button</p>',unsafe_allow_html=True )
    if f'deleting_cols_for_encoding' not in st.session_state: 
        st.session_state[f'deleting_cols_for_encoding']=['Continue','Delete']
    st.selectbox('Delete columns',st.session_state[f'deleting_cols_for_encoding'],key='deleting_cols')
    
    if st.session_state[f'deleting_cols']=='Delete':
        X=X.drop(columns=[col for col in list_missing_cols])
        X
        X.shape
        return X 
    elif st.session_state[f'deleting_cols']=='Continue':
        X
        X.shape
        return X

#to choose how the NaN values will be replaced in every column separately
def imputation(X,X_train,X_valid):
    train=pd.DataFrame()
    valid=pd.DataFrame()
    no_imp_train=pd.DataFrame()
    no_imp_valid=pd.DataFrame()
    count=0
    for column in X.columns:
        if X[column].isnull().sum().sum()==0:
            no_imp_train[column]=pd.DataFrame(X_train[column])
            no_imp_valid[column]=pd.DataFrame(X_valid[column])
        else: 
            st.markdown(f"""<p style="color:Green;font-size:20px;font-weight: bold;">Imputation for column \"{column}\"</p>""",unsafe_allow_html=True )

            f'\"{column}\" is missing {X[column].isnull().sum().sum()} values: '
            if f'options' not in st.session_state:
                st.session_state[f'options']=['No imputer','replace_zero','replace_most_frequent','replace_mean','replace_constant_text','replace_constant_number']
            st.selectbox(f'Choose a replacement instead of NaN for column \"{column}\"',st.session_state[f'options'],key=f'{column}_imput')
            if st.session_state[f'{column}_imput']=='No imputer':
                imputer=None
            elif st.session_state[f'{column}_imput']=='replace_zero':
                imputer=SimpleImputer(strategy='constant',fill_value=0)
            elif st.session_state[f'{column}_imput']=='replace_most_frequent':
                imputer=SimpleImputer(strategy='most_frequent')
            elif st.session_state[f'{column}_imput']=='replace_mean':
                try:
                    imputer=SimpleImputer(strategy='mean')
                except ValueError:
                    st.error('Please choose imputer for text')
            elif st.session_state[f'{column}_imput'] == 'replace_constant_text':
                imputer=SimpleImputer(strategy='constant',fill_value=st.text_input('Fill the constant in if you are intending to replace empty values by a constant',key=f'imp_{column}_{count}'))
                st.markdown('<p style="color:Red;font-size:15px;font-weight: bold;">Choose columns from top to botoom which wished to change to \" constant text\" otherwise the higher already typed will be not saved !</p>',unsafe_allow_html=True )
            elif st.session_state[f'{column}_imput'] == 'replace_constant_number':
                imputer=SimpleImputer(strategy='constant',fill_value=st.number_input('Fill the constant in if you are intending to replace empty values by a constant',key=f'imp_{column}_{count}'))
                st.markdown('<p style="color:Red;font-size:15px;font-weight: bold;">Choose columns from top to botoom which wished to change to \" constant number\" otherwise the higher already typed will be not saved !</p>',unsafe_allow_html=True )
            if imputer is not None:
                try:
                    train_imputer=pd.DataFrame(imputer.fit_transform(X_train[[column]]))
                    valid_imputer=pd.DataFrame(imputer.transform(X_valid[[column]]))
                except (ValueError, AttributeError):
                    st.error('Choose imputer for numbers/text if numerical/categorical column')
                else:
                    train[column]=train_imputer
                    valid[column]=valid_imputer
                count+=1
    train=pd.concat([train.reset_index(drop=True),no_imp_train.reset_index(drop=True)],axis=1)
    valid=pd.concat([valid.reset_index(drop=True),no_imp_valid.reset_index(drop=True)],axis=1)
    return train,valid

# except imputation/replacing there can be still categorical columns which can not be used for encoding (fitting of categorical columns for modelling)
def problematic_categories(X,X_train,X_valid): 
    high_cardinality_cols=[c for c in X.columns if X[c].dtype=='object' and X[c].nunique()>15]
    high_cardinality_cols
    # deleting of columns that content only unique values which are not appropriate for modelling/prediction
    for c in high_cardinality_cols:
        if X_train[c].nunique()==len(X_train.index):
            st.markdown(f"""<p style="color:Red;font-size:15px;">Column \"{c}\" was dropped => completely unique values without any duplicity</p>""",unsafe_allow_html=True )
            X_train.drop(c,axis=1,inplace=True)
            X_valid.drop(c,axis=1,inplace=True)
        else:
            f'Column \"{c}\" has {X_train[c].nunique()} unique values.'
            X_train[c].value_counts()
    # if the column is composed of too many values, these have to be replaced by one class. In this case 'other'. The percentage is specified for both train and valid part and it should be kept the amount of classes still under 10 - visible in clean categories list
            st.markdown(f"""<p style="color:green;font-size:15px;">Set up the threshold (percentage) and the changed column \"{c}\" returns: </p>""",unsafe_allow_html=True )
            #Find the threshold value using the percentage and number of instances in the column
            threshold=st.number_input('Threshold',min_value=0,max_value=100,step=1,key='threshold_input')
            threshold_value=int(threshold/100*len(X_train[c]))
            threshold_value
            #Initialise an empty list for our new minimised categories
            categories_list=[]
            #Create a counter dictionary of the form unique_value: frequency
            counts=Counter(X_train[c])
            #Loop through the category name and its corresponding frequency after sorting the categories by descending order of frequency
            for i,j in counts.most_common():
            #Check if the global sum has reached the threshold value
                if j>=threshold_value:
                    categories_list.append(i)
            #Append the category Other to the list
                else:
                    categories_list.append('other')
            clean_categories_list=[]
            for i in categories_list:
                if i!='other':
                    clean_categories_list.append(i)
    # amount of classes + 'other'
            clean_categories_list
            # handle unknown but in valid:
            transformed_col_list_train=[]
            for el in X_train[c]:
                if el in categories_list:
                    transformed_col_list_train.append(el)
                else:
                    transformed_col_list_train.append('other')
            X_train[c]=transformed_col_list_train

            transformed_col_list_valid=[]
            for el in X_valid[c]:
                if el in categories_list:
                    transformed_col_list_valid.append(el)
                else:
                    transformed_col_list_valid.append('other')
            X_valid[c]=transformed_col_list_valid
            return X_train,X_valid

# encoding of cardinal columns-features
def encoding_process(X,train,valid):
    cat=[c for c in train.columns if train[c].dtype=='object']       
    encoder_df=pd.DataFrame()
    for column in cat: 
        if f'options_encoding' not in st.session_state:
            st.session_state[f'options_encoding']=['No Encoder yet','OrdinalEncoder','OneHotEncoder']
        st.selectbox(f'Choose an Encoder for \"{column}\"',st.session_state[f'options_encoding'],key=f'enc_{column}')
        
        if st.session_state[f'enc_{column}']=='No Encoder yet':
            encoder=None
        elif st.session_state[f'enc_{column}']=='OrdinalEncoder':
            # unknown value which occurs in X_train but not in X_valid should be -1
            encoder=OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1,dtype=int)        
        elif st.session_state[f'enc_{column}']=='OneHotEncoder':
            encoder=OneHotEncoder(handle_unknown='ignore',sparse=False,dtype=int)

        if encoder == None:
            'Choose encoder'
        else:
            enc_result_train=pd.DataFrame(encoder.fit_transform(train[[column]]))    
            train_names={numb:f'{column}_{numb}' for numb in range(len(enc_result_train.columns))}            
            enc_result_train.rename(columns=train_names,inplace=True)
            train=pd.concat([train.reset_index(drop=True),enc_result_train.reset_index(drop=True)],axis=1)
            train.drop(column,axis=1,inplace=True)

            enc_result_valid=pd.DataFrame(encoder.transform(valid[[column]]))
            enc_result_valid.rename(columns=train_names,inplace=True)
            valid=pd.concat([valid.reset_index(drop=True),enc_result_valid.reset_index(drop=True)],axis=1)
            valid.drop(column,axis=1,inplace=True)
    return train,valid 


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
    try:
        regr_model.fit(scaled_X_train, y_train)
        regr_model.fit(scaled_X_valid, y_valid)
    except ValueError:
        lb=LabelBinarizer(neg_label=0)
        y_train=(lb.fit_transform(y_train)).argmax(axis=1)
        y_valid=(lb.transform(y_valid)).argmax(axis=1)
        regr_model.fit(scaled_X_train, y_train)
        regr_model.fit(scaled_X_valid, y_valid)
   
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
    return y_train,y_valid

def cross_score_regr(regr_model, scaled_X_train, scaled_X_valid, y_train,y_valid,cv_regr,scoring,error_score="raise"):
   
    score_train=cross_val_score(regr_model, scaled_X_train,y_train,cv=cv_regr,scoring=scoring,error_score="raise") 
    score_valid=cross_val_score(regr_model, scaled_X_valid,y_valid,cv=cv_regr,scoring=scoring,error_score="raise") 
    
    st.write(f'Score for {scoring}-training data is {score_train}.')
    st.write(f'Score for {scoring}-validation data is {score_valid}.')

def cross_validation_regr(regr_model, scaled_X_train, scaled_X_valid, y_train,y_valid,cv_regr,scoring,return_train_score=True,error_score="raise"):
   
    score_train=cross_validate(regr_model, scaled_X_train,y_train,cv=cv_regr,scoring=scoring,return_train_score=True,error_score="raise") 
    score_valid=cross_validate(regr_model, scaled_X_valid,y_valid,cv=cv_regr,scoring=scoring,return_train_score=True,error_score="raise") 
    
    st.write(f'Score for {scoring}-training data is:')
    st.write({key:value for (key,value) in score_train.items()})
    st.write(f'Score for {scoring}-validation data is:')
    st.write({key:value for (key,value) in score_valid.items()})

# Classification model with classification_report (precision, recall, f1), confusion matrix and accuracy score
def class_modelling(class_model, scaled_X_train, scaled_X_valid, y_train,y_valid,y):
    y_train_name=y.name
    try:
        class_model.fit(scaled_X_train, y_train)
        class_model.fit(scaled_X_valid, y_valid)
    except ValueError:
        lb=LabelBinarizer(neg_label=0)
        y_train=(lb.fit_transform(y_train)).argmax(axis=1)
        y_valid=(lb.transform(y_valid)).argmax(axis=1)
        class_model.fit(scaled_X_train, y_train)
        class_model.fit(scaled_X_valid, y_valid)          
    
    y_train_pred = class_model.predict(scaled_X_train)
    y_valid_pred = class_model.predict(scaled_X_valid)

    # Training data => matplotlib heatmap to compare two confusion matrixs in two graphing libraries
    st.subheader('Report and confusion matrix for training data')
    st.caption('Classification report builds a text report showing the main classification metrics.')
    st.caption('The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives.')
    st.caption('The recall computes the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives.')
    st.caption('Confusion matrix evaluates the accuracy of a classification')
    st.text(classification_report(y_train,y_train_pred))
    
    st.write(confusion_matrix(y_train,y_train_pred))
   
    fig,ax=plt.subplots()
    data=confusion_matrix(y_train,y_train_pred)
    annot=True
    sns.heatmap(data=data,annot=annot,ax=ax,linewidths=1,linecolor='yellow').set(title=f'Confusion_matrix of "{y_train_name}"')
    ax.set_xticklabels(y.unique().tolist())
    ax.set_yticklabels(y.unique().tolist())
    ax.tick_params(top=True, labeltop=True, labelrotation=45)
    st.write(fig)
    
    st.write("Accuracy score for training data:", accuracy_score(y_train, y_train_pred))

    # Valid/Testing data => plotly ff heatmap with yaxis reversed
    st.subheader('Report and confusion matrix for testing/valid data')
    st.text(classification_report(y_valid,y_valid_pred))
   
    st.write(confusion_matrix(y_valid,y_valid_pred))
    
    z2=confusion_matrix(y_valid,y_valid_pred)
    x=y.unique().tolist()
    y=y.unique().tolist()
    z_text2=[[str(round(y,2)) for y in x] for x in z2]
    [X,Y]=np.meshgrid(x,y)
    try:
        Z=np.cos(X/2)+np.sin(Y/4)
        fig=ff.create_annotated_heatmap(Z,annotation_text=z_text2)
    except TypeError: 
        fig=ff.create_annotated_heatmap(z=z2, x=x,y=y,annotation_text=z_text2,colorscale='tealgrn')
    fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',xaxis = dict(title='Predicted value'),yaxis = dict(title='Real value'))
    fig.update_layout(yaxis=dict(autorange='reversed'))
    st.write(fig)

    st.write("Accuracy score for testing/valid data:", accuracy_score(y_valid, y_valid_pred))

    return y_train,y_valid
   



def app():
    def clear_text():
        st.session_state['text']=''
    if st.button('Clear all', on_click=clear_text):
       st.experimental_rerun()

    title=st.text_input(label='Choose title of your app',key='text', max_chars=100,)
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
    'Shape of DataFrame (rows, columns)'
    X.shape

    st.markdown('<p style="color:Green;font-size:20px;font-weight: bold;">Missing values</p>',unsafe_allow_html=True )
    f'The sum of missing values is: {X.isnull().sum().sum()}'
    {col:X[col].isnull().sum() for col in X if X[col].isnull().any()}

    st.markdown('<p style="color:Green;font-size:20px;font-weight: bold;">Rows with missing values</p>',unsafe_allow_html=True )
    percentage_rows=st.number_input('Insert the percentage until the missing values accepted for rows',min_value=0,max_value=100, step=1,key='percentage_rows')
    X=deleting_missing_rows(X,percentage_rows)
    st.markdown('<p style="color:Green;font-size:20px;font-weight: bold;">Columns with missing values</p>',unsafe_allow_html=True )
    percentage_cols=st.number_input('Insert the percentage until the missing values accepted for columns',min_value=0,max_value=100, step=1,key='percentage_cols')
    X=info_about_cols(X,percentage_cols)

    st.markdown('<p style="color:Red;font-size:15px;font-weight: bold;">To drop columns without missing values will be possible below dtypes-table</p>',unsafe_allow_html=True )

    st.markdown('<p style="color:Green;font-size:20px;font-weight: bold;">Dtype-table</p>',unsafe_allow_html=True )
    X.dtypes

    # drop the column if not wished in dataset; IN THIS CASE CATEGORICAL VALUES (NOT NUMERICAL)
    st.markdown('<p style="color:Green;font-size:20px;font-weight: bold;">Drop or keep columns</p>',unsafe_allow_html=True )
    option= st.multiselect("Choose columns which should be processed", list(X.columns), default=list(X.columns))
    X=X[option]
        
    # plotly scatter-matrix
    dimensions = st.multiselect("Scatter matrix dimensions", list(X.columns), default=list(X.columns))
    color = st.selectbox("Color", X.columns)
    opacity = st.number_input('Choose opacity', min_value=0.1, max_value=1.0, value=1.0)

    st.write(px.scatter_matrix(X, dimensions=dimensions, color=color, opacity=opacity))

    # separating X and y from dataset
    st.markdown('<p style="color:Green;font-size:30px;font-weight: bold;">Separating X and y</p>',unsafe_allow_html=True )
    y_column = st.selectbox("Choose a column as your target=y. ", X.columns)
    X.dropna(axis=0,subset=[f'{y_column}'],inplace=True)
    y=X[f'{y_column}']

    st.write('Has your dataset convenient distribution for separation into train and test/valid data?')
    bar_df=X[f'{y_column}'].value_counts()
    st.write(f'The target has {len(bar_df)} categories/classes.')
    
    st.write(px.bar(X, x=X[f'{y_column}'].unique().tolist(),y=bar_df,title='Distribution of 'f"{y_column}",color=bar_df,height=400))
 
    # here you can see if any of two columns have a similarity (PCA?)
    col_selection=st.selectbox('Choose an interesting column beside your target', options=[col for col in X.columns if col!=f'{y_column}'])
    if col_selection in X.columns:
        st.write(px.box(X, x=f'{y_column}', color=f'{y_column}', y=f'{col_selection}', points="suspectedoutliers", notched=False))
    else:
        st.warning('Choose again')
    
    # Processing of missing values
    X.drop([f'{y_column}'],axis=1,inplace=True)

    try:
        X_train, X_valid, y_train, y_valid=train_test_split(X,y,train_size=0.8, test_size=0.2,stratify=y,random_state=42)
    except ValueError:
        # https://github.com/davidsbatista/text-classification/issues/1
        X_train, X_valid, y_train, y_valid=train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=42)
    else:
        X_train, X_valid, y_train, y_valid=train_test_split(X,y,train_size=0.8, test_size=0.2,stratify=y,random_state=42)
      
    st.markdown('<p style="color:Green;font-size:30px;font-weight: bold;">Imputation</p>',unsafe_allow_html=True )
    'X_train'
    X_train
    'X_valid'
    X_valid

    f'The sum of missing values is: {X.isnull().sum().sum()}'
    X_train,X_valid=imputation(X,X_train,X_valid)
    'Imputation of training data'
    X_train
    'Imputation of testing/validation data'
    X_valid

    st.markdown('<p style="color:Green;font-size:30px;font-weight: bold;">Dropping or Encoding before scalling</p>',unsafe_allow_html=True )
    st.markdown('<p style="color:Green;font-size:15px;">1) To drope columns go back to the top</p>',unsafe_allow_html=True )
    st.markdown('<p style="color:Green;font-size:15px;">2) Ordinal Encoder(ordering of the categories)</p>',unsafe_allow_html=True )
    st.markdown('<p style="color:Green;font-size:15px;">3) One-ot Encoding(new columns indicating the presence (or absence) of each possible value in the original data)</p>',unsafe_allow_html=True )

    ### Problematic columns
    st.markdown('<p style="color:Red;font-size:20px;font-weight: bold;">Problematic columns: </p>',unsafe_allow_html=True )
    link='[link](https://towardsdatascience.com/dealing-with-features-that-have-high-cardinality-1c9212d7ff1b)'
    st.markdown(f"""Dealing with features that have high cardinality: {link}""",unsafe_allow_html=True)

    X_train,X_valid=problematic_categories(X,X_train,X_valid) 
    'X_train'
    X_train
    'X_valid'
    X_valid 

    st.markdown('<p style="color:Green;font-size:30px;font-weight: bold;">Encoding of categorical columns</p>',unsafe_allow_html=True )
    impl_X_train,impl_X_valid=encoding_process(X,X_train,X_valid) 
    'X_train'
    impl_X_train
    'X_valid'
    impl_X_valid

    # Scaler
    st.markdown('<p style="color:Green;font-size:30px;font-weight: bold;">Scaler</p>',unsafe_allow_html=True )
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
    
    st.markdown('<p style="color:Green;font-size:30px;font-weight: bold;">Modelling: Regression or Classification</p>',unsafe_allow_html=True )
    
    # Model
    '''if you are using RandomForestRegressor for classification tasks, the "probability" you get might greater than 1, or less than 0. That's not what we expected.
    On the other hand, RandomForestClassifier uses accuracy_score as loss function, while RandomForestRegressor uses r2_score. That's a big difference.'''

    # Regression
    option=st.selectbox('Choose a regression model if avalaible',options=['No regression','LinearRegression','Lasso','SVR','RandomForestRegressor','XGBRegressor'])
    if option=='No regression':
        st.warning('Choose a model if avalaible')
        regr_model=None
    
    elif option=='LinearRegression':
        regr_model=LinearRegression()
        st.write(type(regr_model))


    elif option=='Lasso':
        st.write('For numerical reasons, using alpha = 0 with the Lasso object is not advised. Instead, you should use the LinearRegression object.')
        regr_model=Lasso(alpha=st.number_input('Fill the alpha (the degree of sparsity of the estimated coefficients) in', min_value=0.00001, max_value=10000.0000))

    elif option=='SVR':
        regr_model=SVR()

    elif option== 'RandomForestRegressor':
        regr_model=RandomForestRegressor(
        n_estimators=st.number_input('Fill the n_estimators in',min_value=1, max_value=10000,key=0),
        random_state=st.number_input('Fill the random_state in',min_value=0, max_value=10000,key=1),
        max_depth=st.number_input('Fill the max_depth in',min_value=1, max_value=10000,key=2)
        )

    elif option=='XGBRegressor':
        regr_model=XGBRegressor(
        learning_rate=st.number_input('Fill the learning_rate in',min_value=0, max_value=10000),
        n_estimators=st.number_input('Fill the n_estimators in',min_value=1, max_value=10000,key=3),
        n_jobs=st.number_input('Fill the n_jobs in',min_value=0, max_value=10000),
        random_state=st.number_input('Fill the random_state in',min_value=0, max_value=10000,key=4)
        )

    if regr_model is not None:
        y_train,y_valid=regr_modelling(regr_model, scaled_X_train, scaled_X_valid, y_train,y_valid,)
    else:
        st.write('It is not regression or model is not available')
    
        
    # Classification
    if len(bar_df)>10:
        st.markdown('<p style="color:Red;font-size:30px;font-weight: bold;">Oops, Surely not ideal file for Classification!</p>',unsafe_allow_html=True )
        st.warning(f'The amount of classes is {len(bar_df)}. Possible to change the target.')
        
    else:
        option=st.selectbox('Choose a classification model if avalaible',options=['Classification','DecisionTreeClassifier','KNeighborsClassifier','RandomForestClassifier','SVC','LogisticRegression'])
        
        if option=='Classification':
            st.warning('Choose a model if avalaible')
            class_model=None
        
        elif option=='DecisionTreeClassifier':
            class_model=DecisionTreeClassifier(
            random_state=st.number_input('Fill the random_state in',min_value=0, max_value=10000,key=5),
            max_leaf_nodes=st.number_input('Fill the max_leaf_nodes in',min_value=2, max_value=10000)
            )
            
        elif option=='KNeighborsClassifier':
            class_model=KNeighborsClassifier(n_neighbors=st.number_input('Fill the n_neighbors in',min_value=1, max_value=10000))
                    
        elif option=='RandomForestClassifier':
            class_model=RandomForestClassifier(
            n_estimators=st.number_input('Fill the n_estimators in',min_value=1, max_value=10000,key=6),
            random_state=st.number_input('Fill the random_state in',min_value=0, max_value=10000,key=7),
            max_depth=st.number_input('Fill the max_depth in',min_value=1, max_value=10000,key=8)
            )
            
        elif option=='SVC':
            class_model= SVC()
                
        elif option=='LogisticRegression':
            class_model=LogisticRegression(random_state=st.number_input('Fill the random_state in',min_value=0, max_value=10000,key=9))
        
        #return class_option
        if class_model is not None:
            class_modelling(class_model, scaled_X_train, scaled_X_valid, y_train,y_valid,y)
        else:
            st.write('It is not classification or model is not available')

        class_option=option


    ## CROSS VAL SCORE
    ## Cross_val_score runs single metric cross validation whilst cross_validate runs multi metric. This means that cross_val_score will only accept a single metric and return this for each fold, whilst cross_validate accepts a list of multiple metrics and will return all these for each fold.
        st.markdown('<p style="color:Green;font-size:30px;font-weight: bold;">Cross_val_score</p>',unsafe_allow_html=True )

        st.subheader('Regression')
        cv_regr=st.number_input('Fill the cv (the number of folds) for "cross_val_score" of regression model in', min_value=2,max_value=10000,key=10)
        # https://scikit-learn.org/stable/modules/model_evaluation.html be aware => Scoring goes into the option/def but function must be imported ;)
        st.warning("Don't forget to choose your regression model at first!")
        option=st.selectbox('Choose a scoring for "cross_val_score" of regression model if avalaible',options=['No score','mean_absolute_error', 'mean_squared_error', 'r2_score'])
        if option=='No score':
            st.warning('No score chosen')
            scoring=None
        elif option=='mean_absolute_error':
            scoring='neg_mean_absolute_error'
        elif option=='mean_squared_error':
            scoring='neg_mean_squared_error'
        elif option=='r2_score':
            scoring='r2'

        if regr_model is not None and scoring is not None:
            cross_score_regr(regr_model, scaled_X_train, scaled_X_valid, y_train,y_valid,cv_regr,scoring,error_score="raise")
        else:
            st.write('No cross_val_score for regression chosen.')
    

        st.subheader('Binary classification')
        cv_class=st.number_input('Fill the cv (the number of folds) for "cross_val_score" of classification model in', min_value=2,max_value=10000,key=11)
        st.warning("Don't forget to choose your classification model at first!")
        option=st.selectbox('Choose a scoring for "cross_val_score" of binary classification',options=['No score','average_precision','f1','precision','recall','roc_auc',])
        if class_model is not None and option=='No score':
            st.warning('No cross_val_score for classification chosen.')
        elif class_model==None:
            st.warning('No classification model chosen.')
        elif class_option=='SVC' and option=='roc_auc':
            class_model=globals()['SVC'](probability=True)
            try:
                cross_score_regr(class_model, scaled_X_train, scaled_X_valid, y_train,y_valid,cv_class,option,error_score="raise")
            except ValueError:
                st.warning('Multiclass or multilabel')
        else:
            try:    
                cross_score_regr(class_model, scaled_X_train, scaled_X_valid, y_train,y_valid,cv_class,option,error_score="raise")
            except ValueError:
                st.warning('Multiclass or multilabel')
           
        ### samples do not work yet!
        st.subheader('Multiple classification')
        option=st.selectbox('Choose a scoring for "cross_val_score" of classification model if avalaible for multiclass',options=['No score','accuracy','balanced_accuracy','f1_micro','f1_macro','f1_weighted','f1_samples','precision_micro','precision_macro','precision_weighted','precision_samples','recall_micro','recall_macro','recall_weighted','recall_samples','roc_auc_ovr','roc_auc_ovo','roc_auc_ovr_weighted','roc_auc_ovo_weighted',])
        if class_model is not None and option=='No score':
            st.warning('No cross_val_score for classification chosen.')
        elif class_option==None:
            st.warning('No classification model chosen.')
        elif class_option=='SVC' and (option=='roc_auc_ovr' or option=='roc_auc_ovr_weighted' or option=='roc_auc_ovo' or option=='roc_auc_ovo_weighted'):
            class_model=globals()['SVC'](probability=True)
            try:
                cross_score_regr(class_model, scaled_X_train, scaled_X_valid, y_train,y_valid,cv_class,option, error_score="raise")
            except ValueError:
                st.warning('Binary classification')
        else:
            try:
                cross_score_regr(class_model, scaled_X_train, scaled_X_valid, y_train,y_valid,cv_class,option, error_score="raise")
            except ValueError:
                st.warning('Binary classification')
            
    ## Cross_validate
        st.markdown('<p style="color:Green;font-size:30px;font-weight: bold;">Cross_validation with multiple metric evaluation</p>',unsafe_allow_html=True )

        st.subheader('Regression')
        cv_regr=st.number_input('Fill the cv (the number of folds) for "cross_validate" of regression model in', min_value=2,max_value=10000,key=12)
        # https://scikit-learn.org/stable/modules/model_evaluation.html be aware => Scoring goes into the option/def but function must be imported ;)
        st.warning("Don't forget to choose your regression model at first!")
        options=st.multiselect('Choose a scoring for "cross_validation" of regression model if avalaible',['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']) 
        
        if regr_model is not None: 
            try:
                cross_validation_regr(regr_model, scaled_X_train, scaled_X_valid, y_train,y_valid,cv_regr,scoring=options,return_train_score=True,error_score="raise")
            except ValueError:
                st.write('No cross_validate for regression chosen.')
        else:
            'Either regression model or score was not chosen.'

        st.subheader('Binary classification')
        cv_class=st.number_input('Fill the cv (the number of folds) for "cross_validate" of classification model in', min_value=2,max_value=10000,key=13)
        st.warning("Don't forget to choose your classification model at first!")
        options=st.multiselect('Choose a scoring for "cross_validate" of binary classification',['average_precision','f1','precision','recall','roc_auc',]) 
        if class_option is not None and options==[]: 
            options=None
            st.warning('No cross_validate for classification chosen')
        elif class_option==None:
            st.warning('No classification model chosen.')
        elif class_option=='SVC' and option=='roc_auc':
            scorer=make_scorer(roc_auc_score,needs_proba=True)
            class_model=globals()['SVC'](probability=True)
            try:
                cross_validation_regr(class_model, scaled_X_train, scaled_X_valid, y_train,y_valid,cv_class,scoring=scorer,error_score="raise")
            except ValueError:
                st.warning('Multiclass or multilabel')
        elif class_model is not None and options!=[]:
            try:
                cross_validation_regr(class_model, scaled_X_train, scaled_X_valid, y_train,y_valid,cv_class,scoring=options,return_train_score=True,error_score="raise")
            except ValueError:
                st.warning(' Multiclass or multilabel')
        else:
            try:
                cross_validation_regr(class_model, scaled_X_train, scaled_X_valid, y_train,y_valid,cv_class,scoring=options,return_train_score=True,error_score="raise")
            except ValueError:
                st.warning(' Multiclass or multilabel')
    ### samples do not work yet!
        st.subheader('Multiple classification')
        options=st.multiselect('Choose a scoring for "cross_validate" of classification model if avalaible for multiclass',['accuracy','balanced_accuracy','f1_micro','f1_macro','f1_weighted','f1_samples','precision_micro','precision_macro','precision_weighted','precision_samples','recall_micro','recall_macro','recall_weighted','recall_samples','roc_auc_ovr','roc_auc_ovo','roc_auc_ovr_weighted','roc_auc_ovo_weighted',])
        if class_option is not None and options==[]: 
            options=None
            st.warning('No cross_validate for classification chosen')
        elif class_option==None:
            st.warning('No classification model chosen.')
        elif class_model is not None and options!=[]:
            try:
                cross_validation_regr(class_model, scaled_X_train, scaled_X_valid, y_train,y_valid,cv_class,scoring=options,return_train_score=True,error_score="raise")
            except ValueError:
                st.warning(' Multiclass or multilabel')
        #class_option
        #scorers=[]
        #scorers
        elif class_option=='SVC' and (option=='roc_auc_ovr' or option=='roc_auc_ovr_weighted' or option=='roc_auc_ovo' or option=='roc_auc_ovo_weighted'):
            class_model=globals()['SVC'](probability=True)
            if option=='roc_auc_ovr':
                scorer=make_scorer(roc_auc_score,multi_class='ovr',needs_proba=True)
                scorer
                scorers.append(scorer)
                options.remove(option)
            elif option=='roc_auc_ovr_weighted':
                scorer=make_scorer(roc_auc_score,multi_class='ovr',average='weighted',needs_proba=True)
                scorers.append(scorer)
                options.remove(option)
            elif option=='roc_auc_ovo':
                scorer=make_scorer(roc_auc_score,multi_class='ovo',needs_proba=True)
                scorers.append(scorer)
                options.remove(option)
            elif option=='roc_auc_ovo_weighted':
                scorer=make_scorer(roc_auc_score,multi_class='ovo',average='weighted',needs_proba=True)
                scorers.append(scorer)
                options.remove(option)
        #st.write(scorers)
        st.write(options)
        try:
            cross_validation_regr(class_model, scaled_X_train, scaled_X_valid, y_train,y_valid,cv_class,scorers,error_score="raise")
        except ValueError:
            st.warning('Binary classification')
        else:
            cross_validation_regr(class_model, scaled_X_train, scaled_X_valid, y_train,y_valid,cv_class,scorers,error_score="raise")
            
       ### NEEDS TO BE FINISHED AND TESTED

    st.markdown('<p style="color:Green;font-size:30px;font-weight: bold;">Pipeline</p>',unsafe_allow_html=True )
    
    st.subheader('Regression')
    options_regr=st.multiselect('Choose models for "pipeline" if avalaible',['No regression','LinearRegression','Lasso','SVR','RandomForestRegressor','XGBRegressor'])
    meth_options_regr=st.multiselect('Choose methods for "pipeline" if avalaible',options=['No method chosen','predict', 'predict_log_proba', 'predict_proba','score']) 

    if options_regr==[]: 
        st.write("No regression's model chosen.")
    else:
        for opt in options_regr:
            st.subheader(opt)
            opt_mod=globals()[opt]()
            pipe=Pipeline([(opt,opt_mod)])
            pipe.fit(scaled_X_train, y_train)
            Pipeline(steps=[(opt,opt_mod)])
            for meth in meth_options_regr:
                st.write(meth)
                try:
                    if meth =='score':
                        st.write(eval('pipe.' + meth + '(scaled_X_valid,y_valid)'))
                    else:
                        st.write(eval('pipe.' + meth + '(scaled_X_valid)'))
                except AttributeError:
                    st.markdown(f'<p style="color:Red;font-size:30px;font-weight: bold;">This {opt} object has no attribute "{meth}".</p>',unsafe_allow_html=True )

    st.subheader('Classification')

    if len(bar_df)>10:
        st.markdown('<p style="color:Red;font-size:30px;font-weight: bold;">Oops, Surely not ideal file for Classification!</p>',unsafe_allow_html=True )
        st.warning(f'The amount of classes is {len(bar_df)}. Possible to change the target.')
        
    else:

        options_class=st.multiselect('Choose models for "pipeline" if avalaible',['No classification','DecisionTreeClassifier','KNeighborsClassifier','RandomForestClassifier','SVC','LogisticRegression'])
        meth_options_class=st.multiselect('Choose methods for "pipeline" if avalaible',options=['No method chosen','decision_function','predict','predict_log_proba','predict_proba','score']) 
        if options_class==[]: 
            st.write("No classification's model chosen.")
        else:
            for opt in options_class:
                st.subheader(opt)
                for meth in meth_options_class:
                    st.write(meth)
                    if (meth=='predict_proba' or meth=='predict_log_proba') and opt=='SVC':
                        opt_mod=globals()[opt](probability=True)
                    else:
                        opt_mod=globals()[opt]()
                    pipe=Pipeline([(opt,opt_mod)])
                    try:
                        pipe.fit(scaled_X_train, y_train)
                    except ValueError: # SVC: y should be a 1d array, got an array of shape (,) instead.
                        y_train=np.argmax(y_train,axis=1)
                        pipe.fit(scaled_X_train, y_train)
                    Pipeline(steps=[(opt,opt_mod)])
                    try:
                        if meth =='score':
                            try:
                                st.write(eval('pipe.' + meth + '(scaled_X_valid,y_valid)'))
                            except ValueError: # Classification metrics can't handle a mix of multilabel-indicator and multiclass targets
                                st.write(eval('pipe.' + meth + '(scaled_X_valid,np.argmax(y_valid,axis=1))'))
                        else:
                            st.write(eval('pipe.' + meth + '(scaled_X_valid)'))
                    except AttributeError:
                        st.markdown(f'<p style="color:Red;font-size:30px;font-weight: bold;">This {opt} object has no attribute "{meth}".</p>',unsafe_allow_html=True )
                        
            


if __name__ == "__main__":
    app()

 