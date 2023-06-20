import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import itertools
import time

def linear_regression_fit(X,Y):
    LR=linear_model.LinearRegression(fit_intercept=True)
    LR.fit(X,Y)
    RSS=sklearn.metrics.mean_squared_error(Y,LR.predict(X))*len(Y)
    R_squared=LR.score(X,Y)
    return RSS, R_squared


class data_processor:
    
    def __init__(self, df, target_category):
        self.df = df
        self.target_category = target_category
        try:
            self.target = self.df[target_category]
        except:
            print('target not in dataframe')
        
    #remove null columns from the dataframe entirely
    def remove_null(self):
        for column in self.df:
            if self.df[column].isna().sum()>0.5*len(self.df):
                self.df.drop(column, axis=1, inplace=True)
            elif (self.df[column]==0).sum()>0.5*len(self.df) and len(pd.unique(self.df[column]))>25:
                self.df.drop(column, axis=1, inplace=True)
                
    #fill in na values in columns and remove null columns using the above function
    def process_df(self, special_columns):
        
        self.remove_null()
        to_replace={'na':0}
    
        for column in self.df:
            self.df[column]=self.df[column].fillna('na')
            if column in special_columns:
                self.df[column].replace(to_replace, inplace=True)
                
    def transform_output(self):
        self.target, self.target_lambda=stats.boxcox(self.target)

    #function to view relationship between target variable (SalePrice) and other variables in dataframe
    def view_dataframe(self):
        for column in self.df:
            if column!=self.target_category:
                try:
                    self.df.plot(kind='scatter', x=column, y=target, color='b')
                except Exception as e:
                    print(e, column)
                    
    #function to grab columns that are categorical (defined by how many unique vlues exist in that column)
    def generate_categorical_variables(self, unique_vals):
        self.categorical_list=[]
        self.categorical_df_list=[]
        for column in self.df:
            if len(pd.unique(self.df[column]))<=unique_vals:
                self.categorical_list.append(column)
                temp_df=self.df[column]
                self.df.drop(column, axis=1, inplace=True)
                self.categorical_df_list.append(pd.get_dummies(temp_df,columns=[column],dummy_na=True))
             
    #return data
    def get_df(self):
        return self.df
    
    def get_df_without_target(self):
        self.df.drop(self.target_category,axis=1,inplace=True)
        return self.df
    
    def get_target(self):
        return self.target
    
    def get_categorical_variables(self):
        return self.categorical_list
    
    def get_categorical_variables_df(self):
        return self.categorical_df_list
  

#class for selecting the best model
class model_selector:
    
    def __init__(self, data, variable_combos, init_variable_combos, target):
        self.data = data
        self.variable_combos = variable_combos
        self.init_variable_combos = init_variable_combos
        self.target = target
    
    
        
    def model_subset_iterations(self, view, categorical_df_list, categorical_list, categorical=True,display=True):
    
        variables=[]
        self.R_squared_list=[]
        self.good_variable_combos=[]

        #this section needed for handling categorical variables using dummies which are processed in a separate dataframe
        #the dummy values each take up there own column in a dataframe so this is needed to view the categorical variables
        #as a simple input instead of each dummy variable being its own
        if categorical:

            new_data=[]

            for j in range(self.data.shape[1]):
                new_data.append(self.data.iloc[:,j].to_frame())
                variables.append(self.data.iloc[:,j].name)
            for k in range(len(categorical_df_list)):
                new_data.append(categorical_df_list[k])
                variables.append(categorical_list[k])
            
            #run through all combos from start_combos value to num_combos value and take LR fit score of each
            for i in range(self.init_variable_combos,self.variable_combos+1):    
                iter_time=time.time()
                R_squared=[]
                variable_combos=[]

                for combo in itertools.combinations(new_data,i):
                    tmp_lr=linear_regression_fit(pd.concat(list(combo[:i]),axis=1),self.target)
                    RSS=tmp_lr[0]
                    R_squared.append(tmp_lr[1])

                for combo in itertools.combinations(variables,i):
                    variable_combos.append(combo)

                #sort through the fitted scores and only return the top scores with the corresponding variables
                #total returned determined by 'view'
                sorted_R_squared=sorted(R_squared,reverse=True)
                for val in sorted_R_squared[:view]:
                    val_idx=R_squared.index(val)
                    self.R_squared_list.append(val)
                    self.good_variable_combos.append(variable_combos[val_idx])


                if display:
                    print("Time for iteration %d is %f seconds" % (i,time.time()-iter_time))
                    print("The best combos for this iteration are: ", self.good_variable_combos[view*(i-self.init_variable_combos):view*(i-self.init_variable_combos)+view])
                    print('with R-squared scores: ' ,self.R_squared_list[view*(i-self.init_variable_combos):view*(i-self.init_variable_combos)+view])
                    print('\n')

        if not categorical:

            for i in range(self.init_variable_combos,self.variable_combos+1):    

                iter_time=time.time()
                R_squared=[]
                variable_combos=[]

                for combo in itertools.combinations(new_data,i):
                    tmp_lr=linear_regression_fit(pd.concat(list(combo[:i]),axis=1),desired_value)
                    RSS=tmp_lr[0]
                    R_squared.append(tmp_lr[1])

                for combo in itertools.combinations(variables,i):
                    variable_combos.append(combo)

                sorted_R_squared=sorted(R_squared,reverse=True)
                for val in sorted_R_squared[:view]:
                    val_idx=R_squared.index(val)
                    self.R_squared_list.append(val)
                    self.good_variable_combos.append(variable_combos[val_idx])

                if display:
                    print("Time for iteration %d is %f seconds" % (i,time.time()-iter_time))
                    print("The best combos for this iteration are: ", self.good_variable_combos[view*(i-self.init_variable_combos):view*(i-self.init_variable_combos)+view])
                    print('with R-squared scores: ' ,self.R_squared_list[view*(i-self.init_variable_combos):view*(i-self.init_variable_combos)+view])
                    print('\n')
                    
            
        self.sorted_R_squared_list = sorted(self.R_squared_list, reverse = True)[:view] 
        self.sorted_good_variable_combos = []
        for i in range(view):
            self.sorted_good_variable_combos.append(self.good_variable_combos[self.R_squared_list.index(self.sorted_R_squared_list[i])])
                    
    
    def get_good_scores(self):
        return self.sorted_R_squared_list, self.sorted_good_variable_combos