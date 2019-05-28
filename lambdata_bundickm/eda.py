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
      p_hold = Support.placeholders_present(df[column],placeholders)
      rec = Support.null_rec_lookup(calc,p_hold)

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

    #get unique values and add them to table along with corresponding dtypes
    for col in cols:
      unique_vals = Support.list_to_string(list(df[col].unique()[:unq_limit]))
      if (len(list(df[cols].unique())) > unq_limit): 
        unique_vals += '...'

      table.append([col,str(d_types[col]),num_unique[col],unique_vals])
    #output with tabulate library
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
      val_counts = df[feature].value_counts(normalize=True, dropna=False)
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

  def placeholders_present(column,placeholders):
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
    return list_to_string(list(set(p_holds)))

  
  def null_rec_lookup(null_percent,placeholders=False):
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