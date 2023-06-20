import pandas as pd 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from lr_main import *

#import data and process (using categorical variables)
#2 copies one to modify one to keep as is
df=pd.read_csv(r'C:\Users\Tommy\AppData\Local\Programs\Python\Python37\github\ml-projects\lr\data\train.csv')
orig_df = pd.read_csv(r'C:\Users\Tommy\AppData\Local\Programs\Python\Python37\github\ml-projects\lr\data\train.csv')

def run_lr(df, first_iter_combos, second_iter_combos):
    data_proc = data_processor(df, 'SalePrice')
    data_proc.process_df(special_columns=['LotFrontage','MasVnrArea','GarageYrBlt'])
    data_proc.generate_categorical_variables(25)
    categorical_df_list = data_proc.get_categorical_variables_df()
    categorical_list = data_proc.get_categorical_variables()


    #create model selector object and run lr iterations to get scores
    model_sel = model_selector(data_proc.get_df_without_target(), first_iter_combos, 1, data_proc.get_target())
    model_sel.model_subset_iterations(10, categorical_df_list, categorical_list)

    R_squared_list, good_variable_combos = model_sel.get_good_scores()

    #run it again with the best categories from the first search but with more iterations
    viable_categories = np.unique(good_variable_combos).tolist()
    used_df = df[viable_categories].copy()

    viable_data_proc = data_processor(used_df, 'SalePrice')
    viable_data_proc.generate_categorical_variables(25)
    viable_categorical_df_list = viable_data_proc.get_categorical_variables_df()
    viable_categorical_list = viable_data_proc.get_categorical_variables()

    final_model_sel = model_selector(used_df, second_iter_combos, 1, data_proc.get_target())
    final_model_sel.model_subset_iterations(10, viable_categorical_df_list, viable_categorical_list)
    final_R_squared_list, final_good_variable_combos = final_model_sel.get_good_scores()

    #return the best results
    return final_R_squared_list[0], final_good_variable_combos[0], data_proc.get_target()

#same as run_lr but with the output having a boxcox transform
def run_lr_boxcox(df, first_iter_combos, second_iter_combos):
    data_proc = data_processor(df, 'SalePrice')
    data_proc.process_df(special_columns=['LotFrontage','MasVnrArea','GarageYrBlt'])
    data_proc.transform_output()
    data_proc.generate_categorical_variables(25)
    categorical_df_list = data_proc.get_categorical_variables_df()
    categorical_list = data_proc.get_categorical_variables()


    #create model selector object and run lr iterations to get scores
    model_sel = model_selector(data_proc.get_df_without_target(), first_iter_combos, 1, data_proc.get_target())
    model_sel.model_subset_iterations(10, categorical_df_list, categorical_list)

    R_squared_list, good_variable_combos = model_sel.get_good_scores()

    #run it again with the best categories from the first search but with more iterations
    viable_categories = np.unique(good_variable_combos).tolist()
    used_df = df[viable_categories].copy()

    viable_data_proc = data_processor(used_df, 'SalePrice')
    viable_data_proc.generate_categorical_variables(25)
    viable_categorical_df_list = viable_data_proc.get_categorical_variables_df()
    viable_categorical_list = viable_data_proc.get_categorical_variables()

    final_model_sel = model_selector(used_df, second_iter_combos, 1, data_proc.get_target())
    final_model_sel.model_subset_iterations(10, viable_categorical_df_list, viable_categorical_list)
    final_R_squared_list, final_good_variable_combos = final_model_sel.get_good_scores()

    #return the best results
    return final_R_squared_list[0], final_good_variable_combos[0], data_proc.get_target()

#make final LR model to show correlation between actual values and predicted values
def plot_results(x_train, x_test, y_train, y_test):
    LR = linear_model.LinearRegression(fit_intercept = True)
    LR.fit(x_train,y_train)
    y_pred=LR.predict(x_test)
    plt.scatter(y_test,y_pred)
    plt.plot(y_test,np.poly1d(np.polyfit(y_test, y_pred, 1))(y_test),color='r')
    plt.xlabel('Y_true')
    plt.ylabel('Y_pred')
    plt.title('Y_true vs Y_pred')
    print('\n')
    print('Slope and intercept of bestfit line between y_true and y_pred is %f and %f respectively' % (p[0],p[1]))
    print('The R^2 score of the model is: %f' % (LR.score(x_test,y_test)))
    print('\n')
    plt.show()
    plt.close()
    return

if __name__ == '__main__':
    best_R_squared, best_variable_combos = run_lr(df, 3, 10)
    best_df = orig_df[best_variable_combos].copy()
    #again create a list of dataframes for the categorical variables for final model fitting
    categorical_list=[]
    categorical_df_list=[]
    for column in best_df:
        if len(pd.unique(best_df[column]))<=25:
            categorical_list.append(column)
            temp_df=best_df[column]
            best_df.drop(column, axis=1, inplace=True)
            categorical_df_list.append(pd.get_dummies(temp_df,columns=[column],dummy_na=True))

    new_data=[]
    variables=[]
    for j in range(best_df.shape[1]):
        new_data.append(best_df.iloc[:,j].to_frame())
        variables.append(best_df.iloc[:,j].name)
    for k in range(len(categorical_df_list)):
        new_data.append(categorical_df_list[k])
        variables.append(categorical_list[k])

    x_train, x_test, y_train, y_test = model_selection.train_test_split(pd.concat(new_data[:len(new_data)],axis=1), orig_df['SalePrice'], test_size=0.1,random_state=42)
    plot_results(x_train, x_test, y_train, y_test)

    best_R_squared, best_variable_combos = run_lr_boxcox(df, 3, 10)
    best_df = orig_df[best_variable_combos].copy()
    #again create a list of dataframes for the categorical variables for final model fitting
    categorical_list=[]
    categorical_df_list=[]
    for column in best_df:
        if len(pd.unique(best_df[column]))<=25:
            categorical_list.append(column)
            temp_df=best_df[column]
            best_df.drop(column, axis=1, inplace=True)
            categorical_df_list.append(pd.get_dummies(temp_df,columns=[column],dummy_na=True))

    new_data=[]
    variables=[]
    for j in range(best_df.shape[1]):
        new_data.append(best_df.iloc[:,j].to_frame())
        variables.append(best_df.iloc[:,j].name)
    for k in range(len(categorical_df_list)):
        new_data.append(categorical_df_list[k])
        variables.append(categorical_list[k])

    x_train, x_test, y_train, y_test = model_selection.train_test_split(pd.concat(new_data[:len(new_data)],axis=1), orig_df['SalePrice'], test_size=0.1,random_state=42)
    plot_results(x_train, x_test, y_train, y_test)
    