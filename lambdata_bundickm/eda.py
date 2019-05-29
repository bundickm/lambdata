import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from matplotlib.colors import LinearSegmentedColormap
from tabulate import tabulate

class Reports:
  '''
  Reports for exploratory data analysis
  '''

  def nulls(df,placeholders = [-1,-999,-9999,'None','none','missing',
                                     'Missing','Null','null','?','inf']):
    '''
    report null distribution, any possible placeholders, and 
    simple recommendations

    Input:
    df: Pandas DataFrame object
    placeholders: list of common placeholder values used in place of null. 
                  Report.nulls() is case sensitive ('none' != 'None')

    Output:
    print report to screen
    '''
    null_count = df.isnull().sum().loc
    total = len(df)
    headers = ['Column','Nulls','%Null','Placeholders','Recommendation']
    table = []

    #Iterate through each column and append null details to table
    for column in df.columns:
      calc = null_count[column]/total*100
      null_per = str(calc)+'%'
      p_hold = Support._placeholders_present(df[column],placeholders)
      rec = Support._null_rec_lookup(calc,p_hold)

      table.append([column,null_count[column],null_per,p_hold,rec])
    #output with tabulate library
    print(tabulate(table,headers))


  def describe(df):
    '''
    Simple mod to Pandas.DataFrame.describe() to support Reports.rundown

    Input:
    df: Pandas DataFrame object

    Output:
    print report to screen
    '''
    headers = ['Column']+list(df.describe()[1:].T)
    table = df.describe()[1:].T.reset_index().to_numpy()
    #output with tabulate library
    print(tabulate(table,headers,))


  def type_and_unique(df,unq_limit=10):
    '''
    report data type of all features, number of unique values, and
    some of those values

    Input:
    df: Pandas DataFrame object
    unq_limit: number of unique items from each feature to display
               if unique items is less than unq_limit then all
               items are displayed

    Output:
    print report to screen
    '''
    cols = df.columns
    d_types = list(df.dtypes)
    num_unique = list(df.nunique())
    table = []
    headers=['Column','Type','nUnique','Unique Values']

    for i in range(len(cols)):
      unique_vals = Support.list_to_string(list(df[cols[i]].unique()[:unq_limit]))
      if (len(list(df[cols[i]].unique())) > unq_limit): 
        unique_vals += '...'

      table.append([cols[i],str(d_types[i]),num_unique[i],unique_vals])
    print(tabulate(table,headers))


  def rundown(df):
    '''
    report giving an overview of a dataframe

    Input:
    df: Pandas DataFrame object

    Output:
    print report to screen
    '''
    print('DataFrame Shape')
    print('Rows:',df.shape[0],'    Columns:',df.shape[1])
    print()
    Reports.nulls(df)
    print()
    Reports.describe(df)
    print()
    Reports.type_and_unique(df)
   
   
  def assess_categoricals(df,low_thresh=.05, high_thresh=.51,
                          return_low_violators=False):
    '''
    report for categorical features, highlighting labels in a feature
    that are the majority or extreme minority classifiers

    Input:
    df: Pandas DataFrame object
    low_thresh: float minimum percent distribution desired before binning
    high_thresh: float max percent distribution for majority classifiers
    return_low_violators: bool, if true, include labels below low_thresh
                          as part of report

    Output:
    print report to screen
    '''
    cols = df.select_dtypes(exclude='number').columns
    headers = ['Feature','# Below Thresh','nUnique','High Thresh Violators']
    if return_low_violators == True: headers.append('Low Thresh Violators')
    table = []

    #iterate over all features
    for feature in cols:
      val_counts = df[feature].value_counts(normalize=True)
      low_thresh_count = 0
      low_thresh_violators = []
      high_thresh_violators = []

      #count and record values below low_thresh, and above high_thresh
      for label in val_counts.index:
        if val_counts[label] < low_thresh:
          low_thresh_count += 1
          low_thresh_violators.append(label)
        elif val_counts[label] > high_thresh:
          high_thresh_violators.append(label)

      #append to table based on whether we are returning low_violators
      if return_low_violators == True:
        table.append([feature, low_thresh_count, len(val_counts),
                      high_thresh_violators, low_thresh_violators])
      else:
        table.append([feature, low_thresh_count, 
                      len(val_counts), high_thresh_violators])

    #output with tabulate library
    print(tabulate(table,headers))


class Support:
  '''
  supporting functions for exploratory data analysis, eventually to be 
  broken into Support and Clean
  '''

  def _placeholders_present(column,placeholders):
    '''
    return a list of values that are both in column and placeholders

    Input:
    column: Pandas Series object
    placeholders: a list of values commonly used in place of null

    Output:
    return a list of values that are both in column and placeholders 
    '''
    p_holds = []
    for item in placeholders:
      if len(column.isin([item]).unique())==2:
        p_holds.append(item)
    return Support.list_to_string(list(set(p_holds)))

  
  def _null_rec_lookup(null_percent,placeholders=False):
    '''
    Recommend course of action for handling nulls based on 
    findings from Report.nulls

    Input:
    null_percent: float, percent of a column that is null
    placeholders: bool, whether the column contains placeholders

    Output:
    Return a string recommendation
    '''
    #Temp Recommendations - Switch to MCAR, MAR, MNAR assessment and recs
    #https://www.youtube.com/watch?v=2gkw2T5jAfo&feature=youtu.be
    #https://stefvanbuuren.name/fimd/sec-MCAR.html
    if placeholders:
      return 'Possible Placeholders: Replace and rerun nulls report.'
    elif null_percent == 100:
      return 'Empty Column: Drop column'
    elif null_percent >= 75:
      return 'Near Empty Column: Create binary feature or drop'
    elif null_percent >= 25:
      return 'Partially Filled Column: Assess manually'
    elif null_percent > 0:
      return 'Mostly Filled Column: Impute values'
    else:
      return ''  
    

  def list_to_string(list):
    '''
    helper function to convert lists to string and keep clean code

    Input:
    list: a list

    Output:
    return a string made from list with comma and space separator
    between values.
    '''
    return ', '.join(str(item) for item in list)
  

  def strip_columns(df):
    '''
    helper function to remove leading or trailing spaces from
    all values in a dataframe

    Input:
    df: Pandas DataFrame Object

    Output:
    return a Pandas DataFrame object
    '''
    df = df.copy()

    for col in df.select_dtypes(exclude='number').columns:
      df[col] = df[col].str.strip()

    return df


  def outlier_mask(feature,inclusive=True):
    '''
    creates a mask of the outliers using IQR
    
    Input:
    feature: Pandas Series object containing numeric values
    inclusive: bool, default is True, whether to include values that lie on the
              boundary of becoming an outlier. False will consider the edge
              cases as outliers.

    Output:
    return a Pandas Series object of booleans where True values correspond
    to outliers in the original feature
    '''
    q1 = feature.quantile(.25)
    q3 = feature.quantile(.75)
    iqr = q3-q1
    mask = ~feature.between((q1-1.5*iqr), (q3+1.5*iqr), inclusive=inclusive)
    return mask


  def trimean(feature):
    '''
    calculate the trimean of a numeric feature. Trimean is a measure of the
    center that combines the medians emphasis on center values with the 
    midhinge's attention to the extremes.
    
    Input:
    feature: Pandas Series object containing numeric values
    
    Output:
    return the trimean as a float
    '''
    q1 = feature.quantile(.25)
    q2 = feature.median()
    q3 = feature.quantile(.75)
    
    return ((q1+2*q2+q3)/4)


class Plot:
  def univariate_distribution(df, cols=5, width=20, height=15, 
                              hspace=0.2, wspace=0.5):
    '''
    plot the distribution of all features in a dataframe
    original function found here:
    https://github.com/dformoso/sklearn-classification/blob/master/Data%20Science%20Workbook%20-%20Census%20Income%20Dataset.ipynb

    Input:
    df: Pandas DataFrame object
    cols: number of graphs to display per row
    width: figure width
    height: figure height
    hspace: the amount of height reserved for space between subplots
    wspace: the amount of width reserved for space between subplots

    Output:
    Display n graphs to the screen, where n is the number of features in df
    '''
    #plot settings
    fig = plt.figure(figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                          wspace=wspace, hspace=hspace)
    rows = ceil(float(df.shape[1]) / cols)

    #plot graphs, graph type determined by categoric or numeric feature
    for i, column in enumerate(df.columns):
      ax = fig.add_subplot(rows, cols, i + 1)
      ax.set_title(column)
      if df.dtypes[column] == np.object:
        g = sns.countplot(y=column, data=df)
        substrings = [s.get_text()[:18] for s in g.get_yticklabels()]
        g.set(yticklabels=substrings)
        plt.xticks(rotation=25)
      else:
        g = sns.distplot(df[column])
        plt.xticks(rotation=25)
        
        
  def bivariate_categorical_distribution(df, hue, cols=5, width=20, 
                                         height=15, hspace=0.2, wspace=0.5):
    '''
    Plot a count of the categories from each categorical feature split by hue
    original function found here:
    https://github.com/dformoso/sklearn-classification/blob/master/Data%20Science%20Workbook%20-%20Census%20Income%20Dataset.ipynb
    
    Input:
    df: Pandas DataFrame object
    hue: a categorical feature, likely the target feature
    cols: number of graphs to display per row
    width: figure width
    height: figure height
    hspace: the amount of height reserved for space between subplots
    wspace: the amount of width reserved for space between subplots
    
    Output:
    Display n graphs to the screen, where n is the number of features in df
    '''
    #plot settings
    df = df.select_dtypes(include=[np.object])
    fig = plt.figure(figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                        wspace=wspace, hspace=hspace)
    rows = ceil(float(df.shape[1]) / cols)
    
    #plot each feature's distribution against hue
    for i, column in enumerate(df.columns):
      ax = fig.add_subplot(rows, cols, i + 1)
      ax.set_title(column)
      if df.dtypes[column] == np.object:
        g = sns.countplot(y=column, hue=hue, data=df)
        substrings = [s.get_text()[:10] for s in g.get_yticklabels()]
        g.set(yticklabels=substrings)
        
        
  def correlation_heatmap(df,figsize=(5,5),annot=True):
    '''
    Heatmap of feature correlations of df

    Input:
    df: Pandas DataFrame object
    figsize: tuple of the height and width of the heatmap
    annot: bool, whether to display values inside the heatmap

    Output:
    display heatmap of the feature correlations of df
    '''
    corr = df.corr()

    #generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    #plot it
    f, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, mask=mask, annot=annot, square=True)
    
    
  def null_correlation_heatmap(df,figsize=(5,5),annot=True):
    '''
    heatmap of correlation heatmap of nulls

    Input:
    df: Pandas DataFrame object
    figsize: tuple of the height and width of the heatmap
    annot: bool, whether to display values inside the heatmap

    Output:
    display heatmap of the correlations of nulls in df
    '''
    null_corr = df.isnull().corr()

    #generate a mask for the upper triangle
    mask = np.zeros_like(null_corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    #plot it
    f, ax = plt.subplots(figsize=figsize)
    sns.heatmap(null_corr, mask=mask, annot=annot, square=True)
    
    
  def missingness_map(df,data_name='DataFrame'):
    '''
    graph of the location of all missing values in a dataframe

    Input:
    df: Pandas DataFrame object
    data_name: string for the title of the graph

    Output
    graph of the location of all missing values in a dataframe
    '''
    #map where nulls are in the dataframe
    null_map = df.isnull()

    #create a binary colormap
    myColors = ((0,0,0,1), (1,1,1,1))
    cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))

    #use sns.heatmap and basic cleanup
    ax = sns.heatmap(null_map,vmin=0,vmax=1,cmap=cmap,)
    ax.set(yticks=[])
    ax.set_ylabel('Observations\n(Descending Order)')
    ax.set_xlabel('Features')
    plt.title(f'Missingness Map for {data_name}')

    #Binary color bar with labels
    ax.collections[0].colorbar.set_ticks([.75,.25])
    ax.collections[0].colorbar.set_ticklabels(['Null','Not Null'])

    #display the graph
    plt.show()