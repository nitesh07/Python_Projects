import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Loading a dataframe 
housing_df = pd.read_csv('Housing.csv')

# check basics to understand features and targeted variable
#print(housing_df.head(10))
print(housing_df.info())

# Look into null values 
housing_df.isnull().sum()

# seprate categorical and numerical columns 
categorical_cols = housing_df.select_dtypes(include='object').columns.tolist()
numrical_cols = housing_df.select_dtypes(include='number').columns.tolist()

print('Categorical columns are: \n ',categorical_cols)
print('Numerical columns are :\n ',numrical_cols)

# Check the correlation between price and numeric columns 

corr_matrix = housing_df[numrical_cols].corr()

#plot heatmap 

plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',fmt=".2f")
plt.title('Correlation matrix for numerical features')
plt.show()

## now check categorical variables correlation with price using ANOVA
## Why is ANOVA: the difference in mean price between these categories statistically significant, or just due to random variation?
## Anaova function 
def anova_run(df,target_var,categorical_var):
    """
    The function will return one-way ANOVA for categorical variables
    Args:
    df : Pandas Dataframe 
    target_var: Numeric columns 
    categorical_var: categorical column to group by 
    """
    
    groups = [group[target_var].values for _,group in df.groupby(categorical_var)]
    f_stat , p_val = f_oneway(*groups)
    
    print(f'Anova {target_var} vs. {categorical_var}')
    print(f'f_stat : {f_stat:.2f}')
    print(f'p_val : {p_val:.2f}')
    return f_stat , p_val

# to check the correlation of categorical variable and price 
for column in categorical_cols:
    anova_run(housing_df,'price',column)
    
### finding here is:
### higher F-value means the group means are more spread out 
### P-Value <0.05 means the categorical feature has significant effect on target

def plot_group_means(df,targeted_var,categorical_var):
    """
    Print average target value for each category in categorical variables
    
    Args:
    df : Pandas DataFrame
    targeted_var : Numeric column (e.g., 'price')
    categorical_var : Categorical column (e.g., 'furnishingstatus')
    """
    
    group_stats = df.groupby(categorical_var)[targeted_var].agg(['mean', 'count', 'std']).reset_index()
    
    # calculate standard error mean (SEM = STD/sqrt(count))
    group_stats['sem'] = group_stats['std']/(group_stats['count'])**0.5
    
    plt.figure(figsize=(10,6))
    sns.barplot(
    data = group_stats,
    x=categorical_var,
    y='mean',
    #yerr = group_stats['sem'], # uncomment this to look at standar error 
    palette='Set2')
    
    plt.ylabel(f'Average {targeted_var}')
    plt.xlabel(f'{categorical_var}')
    plt.title(f'{targeted_var} by {categorical_var}')
    plt.show()

# visualize furnishingstatus
plot_group_means(housing_df, 'price','furnishingstatus')

# Now we have identified features by looking at correlation and ANOVA , now we will encode categorical variables 
# now evaluate these vars to check whether they have binary lables, or more then two lables 

# I am using group by counts to have labels and counts
lst = []
for column in categorical_cols:
    lst.append(housing_df.groupby([column])[column].count())
print(lst)

# one hot encoder by dropping the first column to reduce multicollinearity 
ohe_drop = OneHotEncoder(drop='first',sparse_output=False)
housing_df['furnishingstatus']= ohe_drop.fit_transform(housing_df[['furnishingstatus']])

# for binary columns we will use label encoder 
binary_cols = ['mainroad','guestroom','airconditioning','prefarea']

label_encoder = LabelEncoder()
for column in binary_cols:
    housing_df[column] = label_encoder.fit_transform(housing_df[column])

X = housing_df[numrical_cols+binary_cols+['furnishingstatus']] # feature selection 
# new price for better analysis
y = housing_df[['price']]/100000
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=40)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
plt.scatter(y_pred,y_pred, c='r',marker = 's',label = 'Prediction')
plt.scatter(y,y,c='b',marker = 'x',label = 'Actual')
plt.title('Actual vs. Prediction')

# Accuracy Scores
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test,y_pred)

print(f'mse:{mse: .2f} ')
print('r2: ',r2)


