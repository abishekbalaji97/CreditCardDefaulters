# Check the versions of the installed libraries before loading the libraries required for building the model

# Checking Python version
import sys
print('Python: {}'.format(sys.version))

#Checking scipy version
import scipy
print('scipy: {}'.format(scipy.__version__))

# Checking numpy version
import numpy
print('numpy: {}'.format(numpy.__version__))

# Checking matplotlib version
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))

# Checking pandas version
import pandas
print('pandas: {}'.format(pandas.__version__))

# Checking scikit-learn version
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

# Checking seaborn version
import seaborn as sns
print('seaborn: {}'.format(sns.__version__))

# Loading the  libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from scipy import stats

# Use numpy to convert to pandas dataframes to arrays
import numpy as np

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Import train_test_split function from scikit-learn
from sklearn.model_selection import train_test_split

# Load dataset
#names = ['Temperature', 'Precipitation', 'Effective Rainfall', 'Insolation', 'Light usage efficiency','Light interception factor','Wind speed','Humidity','Days after emergence',
#'Efficiency_score']
url=open("default_of_credit_card_clients.csv","r")
dataset = pandas.read_csv(url)

#Change the size of the pandas output screen to accomodate all the columns of a dataset for statistical analysis
pandas.set_option('display.max_rows', 90)
pandas.set_option('display.max_columns', 90)
pandas.set_option('display.width', 70)

#to find the dimensionality of the DataFrame before removing null records
print("\nThe dimensionality of the dataset before removing null records",dataset.shape,"\n")
dataset.columns =['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
       'default payment next month' ]
# head is used to peek at the first 20 records
dataset=df = dataset.iloc[1:]

dataset=dataset.drop("ID", axis=1)
for col in dataset.columns:
    print(col)
dataset.info()
#Printing first 20 records	
print(dataset.head(20))

#Mapping the datatype of each column to the right pandas data type

dataset["LIMIT_BAL"]= dataset["LIMIT_BAL"].astype(float)
dataset["AGE"]=dataset["AGE"].astype(int)
dataset["BILL_AMT1"]=dataset["BILL_AMT1"].astype(int)
dataset["BILL_AMT2"]=dataset["BILL_AMT2"].astype(int)
dataset["BILL_AMT3"]=dataset["BILL_AMT3"].astype(int)
dataset["BILL_AMT4"]=dataset["BILL_AMT4"].astype(int)
dataset["BILL_AMT5"]=dataset["BILL_AMT5"].astype(int)
dataset["BILL_AMT6"]=dataset["BILL_AMT6"].astype(int)
dataset["BILL_AMT6"]=dataset["BILL_AMT6"].astype(int)
dataset["PAY_AMT1"]=dataset["PAY_AMT1"].astype(int)
dataset["PAY_AMT2"]=dataset["PAY_AMT2"].astype(int)
dataset["PAY_AMT3"]=dataset["PAY_AMT3"].astype(int)
dataset["PAY_AMT4"]=dataset["PAY_AMT4"].astype(int)
dataset["PAY_AMT5"]=dataset["PAY_AMT5"].astype(int)
dataset["PAY_AMT6"]=dataset["PAY_AMT6"].astype(int)
dataset["PAY_AMT6"]=dataset["PAY_AMT6"].astype(int)
dataset["PAY_0"]=dataset["PAY_0"].astype(int)
dataset["PAY_2"]=dataset["PAY_0"].astype(int)
dataset["PAY_3"]=dataset["PAY_0"].astype(int)
dataset["PAY_4"]=dataset["PAY_0"].astype(int)
dataset["PAY_5"]=dataset["PAY_0"].astype(int)
dataset["PAY_6"]=dataset["PAY_0"].astype(int)
dataset["SEX"]=dataset["SEX"].astype(int)
dataset["EDUCATION"]=dataset["EDUCATION"].astype(int)
dataset["MARRIAGE"]=dataset["MARRIAGE"].astype(int)
dataset["default payment next month"]=dataset["default payment next month"].astype(int)

#Normalizing the datasets

dataset["LIMIT_BAL"]=dataset["LIMIT_BAL"]/dataset["LIMIT_BAL"].max()
dataset["AGE"]=dataset["AGE"]/dataset["AGE"].max()
dataset["BILL_AMT1"]=dataset["BILL_AMT1"]/dataset["BILL_AMT1"].max()
dataset["BILL_AMT2"]=dataset["BILL_AMT2"]/dataset["BILL_AMT2"].max()
dataset["BILL_AMT3"]=dataset["BILL_AMT3"]/dataset["BILL_AMT3"].max()
dataset["BILL_AMT4"]=dataset["BILL_AMT4"]/dataset["BILL_AMT4"].max()
dataset["BILL_AMT5"]=dataset["BILL_AMT5"]/dataset["BILL_AMT5"].max()
dataset["BILL_AMT6"]=dataset["BILL_AMT6"]/dataset["BILL_AMT6"].max()
dataset["BILL_AMT6"]=dataset["BILL_AMT6"]/dataset["BILL_AMT6"].max()
dataset["PAY_AMT1"]=dataset["PAY_AMT1"]/dataset["PAY_AMT1"].max()
dataset["PAY_AMT2"]=dataset["PAY_AMT2"]/dataset["PAY_AMT2"].max()
dataset["PAY_AMT3"]=dataset["PAY_AMT3"]/dataset["PAY_AMT3"].max()
dataset["PAY_AMT4"]=dataset["PAY_AMT4"]/dataset["PAY_AMT4"].max()
dataset["PAY_AMT5"]=dataset["PAY_AMT5"]/dataset["PAY_AMT5"].max()
dataset["PAY_AMT6"]=dataset["PAY_AMT6"]/dataset["PAY_AMT6"].max()

#Check the data types of each column
#print(dataset.dtypes)
#print("\n")


#print("\n")

# description of quick statistics of dataset
print("Quick statistics of the dataset")
print(dataset.describe(include="all"))
print("\n")

#to print the summary of the DataFrame
print("DataFrame summary")
print(dataset.info(verbose=None, buf=None, max_cols=None, memory_usage=None, null_counts=None))
print("\n")

# class distribution ie. the number of instances of each class are shown below.
print("Number of instances of each class")
print(dataset.groupby('default payment next month').size())
print("\n")

# Features
X=dataset[['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
       'default payment next month']] 

# Labels
y=dataset['default payment next month'] 

# box and whisker plots generation to get short summary of sample and measures of data
#and to spot outliers easily
X.plot(kind='box', subplots=True, layout=(23,23), sharex=False, sharey=False)
plt.show()

# histogram generation
X.hist() 
plt.show()

#Pearson coefficients and p-values of attributes
print("The Pearson coefficients and p-values of each of the attributes are shown below\n")
pearson_coef, p_value = stats.pearsonr(dataset['LIMIT_BAL'], dataset['default payment next month'])
print("The Pearson Correlation Coefficient of LIMIT_BAL is", pearson_coef, " with a P-value of P =", p_value,"\n") 

pearson_coef, p_value = stats.pearsonr(dataset['SEX'], dataset['default payment next month'])
print("The Pearson Correlation Coefficient of SEX is", pearson_coef, " with a P-value of P =", p_value,"\n") 

pearson_coef, p_value = stats.pearsonr(dataset['EDUCATION'], dataset['default payment next month'])
print("The Pearson Correlation Coefficient of EDUCATION is", pearson_coef, " with a P-value of P =", p_value,"\n") 

pearson_coef, p_value = stats.pearsonr(dataset['MARRIAGE'], dataset['default payment next month'])
print("The Pearson Correlation Coefficient of MARRIAGE is", pearson_coef, " with a P-value of P =", p_value,"\n") 

pearson_coef, p_value = stats.pearsonr(dataset['AGE'], dataset['default payment next month'])
print("The Pearson Correlation Coefficient of AGE is", pearson_coef, " with a P-value of P =", p_value,"\n") 

pearson_coef, p_value = stats.pearsonr(dataset['PAY_0'], dataset['default payment next month'])
print("The Pearson Correlation Coefficient of PAY_0 is", pearson_coef, " with a P-value of P =", p_value,"\n") 

pearson_coef, p_value = stats.pearsonr(dataset['PAY_2'], dataset['default payment next month'])
print("The Pearson Correlation Coefficient of PAY_2 is", pearson_coef, " with a P-value of P =", p_value,"\n") 

pearson_coef, p_value = stats.pearsonr(dataset['PAY_3'], dataset['default payment next month'])
print("The Pearson Correlation Coefficient of PAY_3 is", pearson_coef, " with a P-value of P =", p_value,"\n") 

pearson_coef, p_value = stats.pearsonr(dataset['PAY_4'], dataset['default payment next month'])
print("The Pearson Correlation Coefficient of PAY_4 is", pearson_coef, " with a P-value of P =", p_value,"\n") 

pearson_coef, p_value = stats.pearsonr(dataset['PAY_5'], dataset['default payment next month'])
print("The Pearson Correlation Coefficient of PAY_5 is", pearson_coef, " with a P-value of P =", p_value,"\n") 

pearson_coef, p_value = stats.pearsonr(dataset['PAY_6'], dataset['default payment next month'])
print("The Pearson Correlation Coefficient of PAY_6 is", pearson_coef, " with a P-value of P =", p_value,"\n") 

pearson_coef, p_value = stats.pearsonr(dataset['BILL_AMT1'], dataset['default payment next month'])
print("The Pearson Correlation Coefficient of BILL_AMT1 is", pearson_coef, " with a P-value of P =", p_value,"\n")

pearson_coef, p_value = stats.pearsonr(dataset['BILL_AMT2'], dataset['default payment next month'])
print("The Pearson Correlation Coefficient of BILL_AMT2 is", pearson_coef, " with a P-value of P =", p_value,"\n") 

pearson_coef, p_value = stats.pearsonr(dataset['BILL_AMT3'], dataset['default payment next month'])
print("The Pearson Correlation Coefficient of BILL_AMT3 is", pearson_coef, " with a P-value of P =", p_value,"\n")

pearson_coef, p_value = stats.pearsonr(dataset['BILL_AMT4'], dataset['default payment next month'])
print("The Pearson Correlation Coefficient of BILL_AMT4 is", pearson_coef, " with a P-value of P =", p_value,"\n")

pearson_coef, p_value = stats.pearsonr(dataset['BILL_AMT5'], dataset['default payment next month'])
print("The Pearson Correlation Coefficient of BILL_AMT5 is", pearson_coef, " with a P-value of P =", p_value,"\n")

pearson_coef, p_value = stats.pearsonr(dataset['BILL_AMT6'], dataset['default payment next month'])
print("The Pearson Correlation Coefficient of BILL_AMT6 is", pearson_coef, " with a P-value of P =", p_value,"\n")

pearson_coef, p_value = stats.pearsonr(dataset['PAY_AMT1'], dataset['default payment next month'])
print("The Pearson Correlation Coefficient of PAY_AMT1 is", pearson_coef, " with a P-value of P =", p_value,"\n")

pearson_coef, p_value = stats.pearsonr(dataset['PAY_AMT2'], dataset['default payment next month'])
print("The Pearson Correlation Coefficient of PAY_AMT2 is", pearson_coef, " with a P-value of P =", p_value,"\n")

pearson_coef, p_value = stats.pearsonr(dataset['PAY_AMT3'], dataset['default payment next month'])
print("The Pearson Correlation Coefficient of PAY_AMT3 is", pearson_coef, " with a P-value of P =", p_value,"\n")

pearson_coef, p_value = stats.pearsonr(dataset['PAY_AMT4'], dataset['default payment next month'])
print("The Pearson Correlation Coefficient of PAY_AMT4 is", pearson_coef, " with a P-value of P =", p_value,"\n")

pearson_coef, p_value = stats.pearsonr(dataset['PAY_AMT5'], dataset['default payment next month'])
print("The Pearson Correlation Coefficient of PAY_AMT5 is", pearson_coef, " with a P-value of P =", p_value,"\n")

pearson_coef, p_value = stats.pearsonr(dataset['PAY_AMT6'], dataset['default payment next month'])
print("The Pearson Correlation Coefficient of PAY_AMT6 is", pearson_coef, " with a P-value of P =", p_value,"\n")


# scatter plot matrix generation 
#scatter_matrix(X)
#plt.show()

#Finding the correlation between the different input variables
print("The correlation between different input variables is given below")
print(dataset[['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
       'default payment next month']].corr())
print("\n")

# Labels are the values we want to predict
labels = np.array(dataset['default payment next month'])
# Remove the labels from the features
# axis 1 refers to the columns
features= dataset.drop('default payment next month', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)


# Split dataset into training set and test set
# 78% training and 22% test set splitting is achieved with the above line of code
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.22,random_state=42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

#Create a Gaussian Classifier using the below command
clf=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(train_features,train_labels)
y_pred=clf.predict(test_features)
print("\nAccuracy:",metrics.accuracy_score(test_labels, y_pred))
print("Precision:",metrics.precision_score(test_labels, y_pred))
print("Recall:",metrics.recall_score(test_labels, y_pred))
print("F-Measure:",metrics.f1_score(test_labels, y_pred))
print("\n")

