# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 00:54:51 2022

@author: kashy
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Data.csv')

'''
Index(['YearStart', 'YearEnd', 'LocationAbbr', 'LocationDesc', 'Datasource',
       'Class', 'Topic', 'Question', 'Data_Value_Unit', 'Data_Value_Type',
       'Data_Value', 'Data_Value_Alt', 'Data_Value_Footnote_Symbol',
       'Data_Value_Footnote', 'Low_Confidence_Limit', 'High_Confidence_Limit ',
       'Sample_Size', 'Total', 'Age(years)', 'Education', 'Gender', 'Income',
       'Race/Ethnicity', 'GeoLocation', 'ClassID', 'TopicID', 'QuestionID',
       'DataValueTypeID', 'LocationID', 'StratificationCategory1',
       'Stratification1', 'StratificationCategoryId1', 'StratificationID1'],
      dtype='object')
'''

df.loc[:,['Topic','Class','Question','ClassID','TopicID','QuestionID']]

def oneonone(df,c1,c2):
    dic = {}
    for row in df.iterrows():
        row = row[1]
        the_tuple = row[c2]
        if(row[c1] not in dic):
            dic[row[c1]] = the_tuple
        else:
            if(the_tuple != dic[row[c1]]):
                return False
    return True

print(oneonone(df,'Topic' , 'Class'))
print(oneonone(df,'Class' , 'Topic'))
print(oneonone(df,'Class' , 'ClassID'))
print(oneonone(df,'ClassID' , 'Class'))
print(oneonone(df,'Topic' , 'TopicID'))
print(oneonone(df,'TopicID' , 'Topic'))
print(oneonone(df,'LocationDesc' , 'LocationID'))
'''
print(oneonone(df,'Topic' , 'Class'))
print(oneonone(df,'Topic' , 'Class'))
print(oneonone(df,'Topic' , 'Class'))
print(oneonone(df,'Topic' , 'Class'))'''


'''
dic = {}
for row in df.iterrows():
    row = row[1]
    the_tuple = (row['Class'] , row['ClassID'],row['TopicID'],)
    if(row['Topic'] not in dic ):
        dic[row['Topic']] = the_tuple
    else:
        if(the_tuple != dic[row['Topic']]):
            print('They are not one on one')
            break
            
    '''
'''
Part 1 results:
The dataset is huge - it has 10976 rows and 33 columns. 9 of the columns don't have any null values while the rest have either 1 or multiple null values. 1 column,Data_Value_Unit has no data, so it can be safely dropped.

LocationAbbr, LocationDesc,Topic, Data_value_alt, QuestionID, StratificationCategoryId1, StratificationID1  are redundant columns. 
We see that Year End and Year start are one-on-one.
Upon Description, we find that the columns 'Data_Value_Type' , 'Datasource', only have 1 unique value and therefore are irrelavent. 

'Data_Value_Footnote' is present wherever there isn't any data. 'Data_Value_Footnote_Symbol' accurately describes this with a tilde, so we don't need the former column

Upon Analysis, we find that the columns 'Topic','Class','ClassID','TopicID' all have the same information. So , we can just use 1 column. I chose Topic. Stratefication and category IDs are also redundant
'''

'''
The Columns Age, Education, Gender, Income, Race/Ethnicity have been aggregated intop 2 columns - StratificationCategory1 and Stratification1. So, we can retain these columns and Delete the rest.
'''
    
df = df.drop(['YearEnd','Data_Value_Unit' , 'LocationAbbr','LocationDesc','Class','ClassID','TopicID','Data_Value_Alt','Data_Value_Footnote','Data_Value_Type','QuestionID','DataValueTypeID','Datasource','Total','StratificationCategoryId1', 'StratificationID1'],axis = 1)
#We can delete the records that do not have data. 
df = df[df['Data_Value_Footnote_Symbol']!='~'].drop(['Data_Value_Footnote_Symbol'],axis=1)

df = df.drop(['Age(years)', 'Education', 'Gender', 'Income',
       'Race/Ethnicity'],axis=1)

df['LocationID'].value_counts()

df.columns[df.isnull().any()]
'''
After We remove all the un-needed columns, we find that the only column with null values is GeoLocation. 
This is because there's no values for the location- USA
'''



df_num = df.select_dtypes(include = ['float64', 'int64'])
df_num_corr = df_num.corr()['Data_Value'][:-1] # -1 means that the latest row is SalePrice
top_features = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False) #displays pearsons correlation coefficient greater than 0.5
print("There is {} strongly correlated values with SalePrice:\n{}".format(len(top_features), top_features))

'''
Only Low_Confidence_Limit and High_Confidence_Limit are highly correlated witht he data. 
However, we don't need to plot these values are we already know they would be correlated from prior data description.
These columns are also not useful for our output - Data_Value.'''

plt.style.use('ggplot')

plt.title('Relation between data_Value and Confidence limits')
plt.xlabel('Data_Value')
plt.ylabel('Data_Value, High and low confidence')
plt.scatter(x=df['Data_Value'],y=df['Data_Value'],s=100,c='red',alpha=0.6,marker='o')
plt.scatter(x=df['Data_Value'],y=df['Low_Confidence_Limit'],s=100,c='blue',alpha=0.5,marker='h')
plt.scatter(x=df['Data_Value'],y=df['High_Confidence_Limit'],s=100,c='green',alpha=0.5,marker='h')
 
plt.legend(loc='upper right')
 
 
plt.tight_layout()
 

df.dtypes
'''The datatypes are int, float and objects.'''
 #Lets now Examine the output column
import seaborn as sns
Untransformed = sns.distplot(df['Data_Value'])
print("Skewness: %f" % df['Data_Value'].skew())

'''
The range of skewness for a fairly symmetrical bell curve distribution is between -0.5 and 0.5;
 moderate skewness is -0.5 to -1.0 and 0.5 to 1.0; 
 and highly skewed distribution is < -1.0 and > 1.0. 
 In our case, we have ~1.7, so it is considered highly skewed data.

The plot shows a very symmetrical distribution centered about the mean on 31.15.
The Skewness is 0.45m which shows that its fairly normally distributed.
There is no need to apply log transformation.
'''
Untransformed = sns.distplot(df['Sample_Size'])

'''
We see that sample sizes are very varied - let's see if this incurs any bias towards the data_value
'''
import matplotlib.pyplot as plt 
#plt.plot(df['Sample_Size'],df['Data_Value'],'o')
plt.style.use('ggplot')
color = df['Sample_Size'].apply(lambda x: 'blue' if x >20000 else 'red')
plt.title('Relation between data_Value and Sample_Size')
plt.xlabel('Data_Value')
plt.ylabel('Sample_Size')
plt.scatter(x=df['Data_Value'],y=df['Sample_Size'],s=100,c=color,alpha=0.6,marker='x')

plt.legend(loc='upper right')


plt.tight_layout()

min(df['Sample_Size']) , max(df['Sample_Size'])
'''
There is an interesting observation - as the sample size increases, the values of Data get narrower
This is the probably the work of the law of large numbers.
The values of Sample Size vary from 50 all the way to 476,876. 
We can choose to delete records with a minimum sample size threshold for more confident results. 


'''

#Handling Duplicates

duplicate = df[df.duplicated()]
duplicate
'''
We find that there are no duplicate records in our data
'''

#Feature Scaling
'''
We find that we don't require feature scaling for our data'
Data_Value is the only true number column/
'''

#Handling Outliers

#1. Univariate analysis

sns.boxplot(x=df['Data_Value'])
'''
Data_Value has outliers - lot of values are beyond the maximum and a few are below minimum. 
But these are real data points and we want to preserve them. 
'''
sns.boxplot(x=df['Sample_Size'])

#2. Bivariate analysis. 
value_per_loc = df.plot.scatter(x='LocationID',
                      y='Data_Value')
value_per_loc = df.plot.scatter(x='YearStart',
                      y='Data_Value')
#There is no evidence that obseity changed with time. 
plt.scatter(x=df['High_Confidence_Limit']-df['Low_Confidence_Limit'] , y=df['Sample_Size'])



plt.style.use('ggplot')
color = df['Sample_Size'].apply(lambda x: 'green' if x >20000 else 'gold')
plt.title('Relation between Sample Size and Confidence Interval')
plt.xlabel('Sample Size')
plt.ylabel('Confidence Interval')
plt.scatter(x=df['Sample_Size'],y=df['High_Confidence_Limit']-df['Low_Confidence_Limit'],s=100,c=color,alpha=0.6,marker='x')
plt.legend(loc='upper right')
plt.tight_layout()



'''
There are no outliers, but;
Again, we see that the confidence interval gets narrower as we increase the sample size.
The correlation is so strong that it can be wise to delete all the low sample size columns.
However, there are 47,442 data points with sample size <20,000 and just 904 points with sample size > 20,000.
So, we don't  want to lose all that information. We can maintain a seperate dataframe for high sample size and see the plots in that dataframe. 
'''
df_high_sample_size = df[df['Sample_Size']>20000]

plt.plot(df_high_sample_size['Sample_Size'],df_high_sample_size['Data_Value'],'o')

plt.scatter(x=df_high_sample_size['High_Confidence_Limit ']-df_high_sample_size['Low_Confidence_Limit'] , y=df_high_sample_size['Sample_Size'])

#Z Score analysis
import scipy.stats 
df['Data-Value'] = scipy.stats.zscore(df['Data_Value'])
df[['Data_Value','Data-Value']].describe().round(3)
'''
As tha max value is beyond Z score +3, it is mathematical proof that we have outliers.
'''
sns.pairplot(df)

###########################################################################
"Let's Start the stratification column"
"Here, we are doing eda to understand how data_Value is related to all statifications"
df_Income = df[df['StratificationCategory1']=='Income']
df_Income.groupby('Stratification1').mean()['Data_Value']
'''
We observe, data value doesn't change wrt salary.
'''

df_age = df[df['StratificationCategory1']=='Age (years)']
sns.set_context('talk')
sns.pairplot(df_age, hue='Stratification1')

df_gender = df[df['StratificationCategory1']=='Gender']
sns.histplot(data = df_gender,x ='Stratification1' , y='Data_Value')



sns.set_context('talk')
sns.pairplot(df_Income, hue='Stratification1')

'''
Hypothesis:
    1. Obsesity is not related to the state
    2. Obesity is not related to time
    3. Obesity isnot related to Income
    4. Obesity is not related to Gender
    

1.NULL HYPOTHESIS:
Obesity is related to the state from chance.
Alternate Hypothesis: 
 There is a correlation between state and obesity
 significance value = 0.05
 alpha/2 = 0.025
 '''
from scipy import stats

female=df_gender[df_gender['Stratification1']=="Female"]['Data_Value']
male=df_gender[df_gender['Stratification1']=="Male"]['Data_Value']


alpha=0.05
t_value1, p_value1 = stats.ttest_ind(female,male)
print("t_value1 = ",t_value1, ", p_value1 = ", p_value1)

if p_value1 <alpha:
    print("Conclusion: since p_value {} is less than alpha {} ". format (p_value1,alpha))
    print("Reject the null hypothesis that there is no difference between obesity of females and obesity of males.")
    
else:
    print("Conclusion: since p_value {} is greater than alpha {} ". format (p_value1,alpha))
    print("Fail to reject the null hypothesis that there is a difference between obesity of females and obesity of males.")
    
