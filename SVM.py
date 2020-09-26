import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.multioutput import MultiOutputRegressor

import hyperopt.hp as hp

from hyperopt import fmin, tpe, Trials,STATUS_OK

import numpy as np

from sklearn.svm import SVC

from sklearn import metrics

from hyperopt import space_eval


#Reading the file through pandas

vehicle=pd.read_excel("vehicle.xlsx")


#function for imputting missing values

def impute_nan(vehicle,y_label='class'):
   
    ''' Using Multioutput regressor to impute missing values'''
   
    x_columns=list(vehicle.columns)
   
    x_columns.remove(y_label)
   
    x_columns=pd.Index(x_columns)
   
    nan_cols=x_columns[vehicle[x_columns].isnull().any()].tolist()

    non_nan_cols=x_columns[~vehicle[x_columns].isnull().any()].tolist()
   
    notnans = vehicle[nan_cols].notnull().all(axis=1)
   
    df_notnans = vehicle[notnans]
   
    X_train, X_test, y_train, y_test = train_test_split(df_notnans[non_nan_cols], df_notnans[nan_cols],
                                                    train_size=0.75,
                                                    random_state=4)
   
    regr_multirf = MultiOutputRegressor(RandomForestRegressor(max_depth=30,
                                                          random_state=0))
    regr_multirf.fit(X_train, y_train)

    score = regr_multirf.score(X_test, y_test)
   
    print("The prediction score of the imputation on the test data is {:.2f}%".format(score*100))

    vehicle_imputed = vehicle.loc[~notnans].copy()
   
    vehicle_imputed[nan_cols] = regr_multirf.predict(vehicle_imputed[non_nan_cols])
   
    vehicle_non_nans_rows=vehicle.loc[notnans].copy()
   
    return pd.concat([vehicle_non_nans_rows,vehicle_imputed], ignore_index=True)


   
# function for dimensionality reduction using PCA  
   
def dim_red(dataframe,method='PCA',n_dims=9,return_model=False):
       
    if method=='PCA':
   
        model= Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=n_dims))])
       
 # SVM was slow so doing scaling and then PCA using sklean Pipeline
       
    X=model.fit_transform(dataframe.copy())
   
    if  return_model:
       
        return model,pd.DataFrame(X,columns=['PC_'+str(x) for x in range(int(n_dims))])
       
    else:
       
        return pd.DataFrame(X,columns=['PC_'+str(x) for x in range(int(n_dims))])



def objective_function(params):
   
    '''
    This function optimizes the params passed for maximum f1-score
    '''
   
    X=params['X']
    y=params['y']
    n_dims=int(params['n_dims'])
   
    del params['X']
    del params['y']
    del params['n_dims']
   
    clf = SVC(**params)    
   
    print(n_dims)
   
    X_red=dim_red(X.copy(),'PCA',n_dims)
   
    print('PCA ran sucessfully ! ')
   
    mean_f1 = 0.0
   
   
     # repeat the CV procedure 10 times to get more precise results
   
    for i in range(10):
       
       
       
    # for each iteration, randomly hold out 20% of the data as CV set
       
       
        X_t, X_cv, y_t, y_cv = train_test_split(
        X_red, y, test_size=.20, random_state=i*2)
   
        # train model and make predictions
       
        clf.fit(X_t, y_t)
        preds = clf.predict(X_cv)
   
        # compute AUC metric for this CV fold
       
        f1 = metrics.f1_score(y_cv, preds,average='weighted')
        print ("F1 (fold %d/%d): %f" % (i + 1, 4, f1))
        mean_f1 += f1
   
   
    return {'loss': -((mean_f1)/10, 'status': STATUS_OK}


#Parameter tuning using hyperopt
   
def tune_params(X_train,y_train):
   
    space={
        'kernel': hp.choice('kernel', ['linear','rbf', 'poly','sigmoid']),
        'degree': hp.uniform('degree', 0,5),
        'gamma': hp.loguniform('gamma', -8,1),
        'C': hp.loguniform('C',1,4),
        'n_dims':hp.quniform('n_dims',1,18,1),
        'X':X_train,
        'y':y_train
        }
   
   
    trials = Trials()
    best_param = fmin(objective_function,space=space, algo=tpe.suggest,
                      max_evals=100, trials=trials, rstate= np.random.RandomState(1))
   
    n_dims=int(best_param['n_dims'])
   
    if 'n_dims' in best_param.keys():del best_param['n_dims']
    if 'X' in best_param.keys():del best_param['X']
    if 'y' in best_param.keys():del best_param['y']
   
    del space['n_dims']
    del space['X']
    del space['y']
   
   
    return space_eval(space, best_param),n_dims

   
# main function
   
def main():
   
    x_cols=['circularity',
     'compactness',
     'distance_circularity',
     'elongatedness',
     'hollows_ratio',
     'max.length_aspect_ratio',
     'max.length_rectangularity',
     'pr.axis_aspect_ratio',
     'pr.axis_rectangularity',
     'radius_ratio',
     'scaled_radius_of_gyration',
     'scaled_radius_of_gyration.1',
     'scaled_variance',
     'scaled_variance.1',
     'scatter_ratio',
     'skewness_about',
     'skewness_about.1',
     'skewness_about.2']
   
    from sklearn.preprocessing import LabelEncoder
   
    le=LabelEncoder()
   
    y_label='class'
   
    data=impute_nan(vehicle,y_label)
   
    X_train, X_test, y_train, y_test = train_test_split(data[x_cols], le.fit_transform(data[y_label]) ,
                                                        train_size=0.8,
                                                        random_state=4)
   
    model_params,n_dims=tune_params(X_train,y_train)
   
   
    final_model=SVC(**model_params)
   
    pca_pipe,X_train_red=  dim_red(X_train,'PCA',n_dims=n_dims,return_model=True)
   
    final_model.fit(X_train_red,y_train)
   
    X_test_red =  pca_pipe.transform(X_test)    
   
    predictions=le.inverse_transform(final_model.predict(X_test_red))
   
    actuals=le.inverse_transform(y_test)
   
    print('SVM confusion matrix for 10% hold out set')
    print('---------------------------------------------------------------')
    print (pd.DataFrame(metrics.confusion_matrix(actuals, predictions),columns=\
                        ['predicted_'+str(x) for x in le.classes_],index=[ 'actual_'+str(x) for x in le.classes_]))
   
   
    print('SVM classification report hold out set')
    print('---------------------------------------------------------------')
    print (metrics.classification_report(actuals, predictions))
   
   
if __name__=='main':
    main()