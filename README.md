# linear-regression-housing-sales

Linear Regression project to predict the sale price of houses based on a large
number of variables. The project involved parsing through the data to see
which were viable variables to be used for prediction.

I then went through a process of model prediction which involved combining 
the variables and running a linear regression fit. I first used a smaller number
of combinations (i.e. combining 3 variables at a time) to reduce calculation time
and took the variables from the top 20 R-squared scores.

Then I could run the same process with larger combinations of variables without
sacrificing efficiency.

lr_main.py includes the classes for data processing and model selection.

lr_run.py runs the model selection process given certain paramters such as
number of combinations and produces a plot comparing y_true with y_pred
using the best combination of variables.

plot_dataframe.py produces plots of given categories vs. sale price and outputs
it to a docx file. Example given below:
[sample_image_doc.docx](https://github.com/woods0813/linear-regression-housing-sales/files/11804745/sample_image_doc.docx)


The plot below shows one of the best fits, using 19 of the variables and a boxcox transformation on the output:
                     
![LR_best_fit](https://github.com/woods0813/linear-regression-housing-sales/assets/114941826/4528d20d-8ff8-4939-afca-fff5048a2b23)


*Some errors may be encountered with the dataframe being modified throughout the process, df.copy() did not seem to be doing the trick, 
so I created multiple copies of the dataframe with pd.read_csv
