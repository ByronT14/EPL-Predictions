import pandas as pd

def lookupclean(lkup,newFile,newFileName,keepName,joinName):
    '''
    Function to add a different variation of a team name to a file using a look-up file. 
    lkup: Name of the look-up file - contains multiple variations of team names as per each source
    newFile: Name of the file to have the common team name added to
    newFileName: Variable that contains the team name in the new file - to be joined on to the lookup file
    keepName: The variable from the lookup table to keep, which will be used for joining on to other tables
    joinName: Variable that from the lookup table which will be joined on to the new file
    '''
    import pandas as pd
    output = lkup[[keepName,joinName]].copy()
    output = pd.merge(output,newFile,left_on=joinName,right_on=newFileName)
    output = output.drop([joinName,newFileName],axis=1)
    return output

# def home_teams(df,keep):
#     outdf = df.copy()
#     outdf = outdf.query("f_HmGame==1")
#     outdf = outdf[keep]
#     outdf=outdf.rename(columns={'f_Team': 'HomeTeam'})
#     return outdf

# def away_teams(df,keep):
#     outdf = df.copy()
#     outdf = outdf.query("f_HmGame==0")
#     outdf = outdf[keep]
#     outdf=outdf.rename(columns={'f_Team': 'AwayTeam'})
#     return outdf

# def fixture_scores(df,score_keep_vars):
#     outdf = df.copy()
#     outdf = outdf.query("f_HmGame==1")
#     outdf = outdf[score_keep_vars]
#     outdf=outdf.rename(columns={'f_Goals': 'HomeGoals',
#                                 'f_Goals Conceded': 'AwayGoals'})
#     return outdf

# def join_fixtures(home,away,scores,merge_vars):
#     outdf= pd.merge(home,away,on=merge_vars)
#     outdf = pd.merge(outdf,scores,on=merge_vars)
#     return outdf
# def final_outcome(fixtures,homegoals,awaygoals):
#     outdf = fixtures.copy()
#     outdf["Result"] = np.where(outdf[homegoals]>outdf[awaygoals],"H",
#                               np.where(outdf[homegoals]<outdf[awaygoals],"A","D"))
#     return outdf

def ema_no_reset(stats, span,feature_cols):
    '''
    Calculates an exponential moving average for each team for the time span required.
    Does not reset after each season
    stats: Dataframe containing Team statistics
    span: Window for which we want to calculate moving average
    feature_cols: list of columns for which we want to create a moving average
    '''
    import pandas as pd
    ema_features = stats[['matchid', 'f_Team', 'gameweek','season','f_HmGame','fixture', 'matchdate','gw_no']].copy()
    for feature_name in set(feature_cols):
        #print(feature_name)
        feature_ema = (stats.groupby(['f_Team'])[feature_name]  # Calculate the EMA
                       .transform(lambda row: row.ewm(span=span, min_periods=2)
                                  .mean()
                                  .shift(1)
                                 ))
        ema_features[feature_name] = feature_ema
    #create some ratios
    ema_features['f_netxg'] = ema_features['f_US xG']-ema_features['f_US xG Conceded']
    ema_features['f_xGRatio'] = ema_features['f_US xG']/(ema_features['f_US xG']+ema_features['f_US xG Conceded'])
    ema_features['f_ShotsRatio'] = ema_features['f_Goal Attempts']/(ema_features['f_Goal Attempts']+ema_features['f_Shots Conceded'])
    ema_features['f_ShOnTargetRatio'] = ema_features['f_Shots On Target']/(ema_features['f_Goal Attempts'])
    return ema_features

#exponentially weighted averages - however this one won't reset after each season
def moving_average_no_reset(stats,span,feature_cols):
    '''
    Calculates an standard moving average for each team for the time span required.
    Does not reset after each season
    stats: Dataframe containing Team statistics
    span: Window for which we want to calculate moving average
    feature_cols: list of columns for which we want to create a moving average
    '''
    ema_features = stats[['matchid', 'f_Team', 'gameweek','season','f_HmGame','fixture', 'matchdate','gw_no']].copy()
    for feature_name in set(feature_cols):
        #print(feature_name)
        feature_ema = (stats.groupby(['f_Team'])[feature_name]  # Calculate the EMA
                       .transform(lambda row: row.rolling(window=span, min_periods=2)
                                  .mean()
                                  .shift(1)
                                 ))
        ema_features[feature_name] = feature_ema
    #create some ratios
    ema_features['f_netxg'] = ema_features['f_US xG']-ema_features['f_US xG Conceded']
    ema_features['f_xGRatio'] = ema_features['f_US xG']/(ema_features['f_US xG']+ema_features['f_US xG Conceded'])
    ema_features['f_ShotsRatio'] = ema_features['f_Goal Attempts']/(ema_features['f_Goal Attempts']+ema_features['f_Shots Conceded'])
    ema_features['f_ShOnTargetRatio'] = ema_features['f_Shots On Target']/(ema_features['f_Goal Attempts'])
    return ema_features

def restructure_ema(ema_df, merge_features,ema_vars):
    import pandas as pd
    stats_restructured_home = (ema_df.query('f_HmGameFut== 1')
                           .rename(columns ={'f_Team':'Team'})).copy()
    stats_restructured_home = (stats_restructured_home[merge_features+ema_vars]
                              .rename(columns={col: col + '_Home' for col in ema_df.columns if col in ema_vars}))
    stats_restructured_away = (ema_df.query('f_HmGameFut== 0')
                           .rename(columns ={'f_Team':'Team'})).copy()
    stats_restructured_away = (stats_restructured_away[merge_features+ema_vars]
                              .rename(columns={col: col + '_Away' for col in ema_df.columns if col in ema_vars}))
    stats_restructured_all = stats_restructured_home.merge(stats_restructured_away, on=merge_features, how='inner') 
    return stats_restructured_all

def create_model_base(df,avg_method, span,ema_features, target_goals
                    ,merge_vars, calc_vars,single_line):
    import pandas as pd
    ema_vars = ema_features + calc_vars
    ema_output = avg_method(df, span,ema_features)
    ema_target = pd.merge(ema_output, target_goals, on = ['f_Team','matchid']).dropna() ## this contains our target variable
    ema_single_line = restructure_ema(ema_target,merge_vars,ema_vars)
    ema_results = pd.merge(single_line, ema_single_line, on=merge_vars,how="inner")
    return ema_results

def create_model_base_scaled(df,avg_method, span,ema_features,scaler, target_goals
                    ,merge_vars, calc_vars,single_line):
    #import pandas as pd
    ema_vars = ema_features + calc_vars
    ema_output = avg_method(df, span,ema_features)
    ema_output[ema_features] = scaler.fit_transform(ema_output[ema_features])
    ema_target = pd.merge(ema_output, target_goals, on = ['f_Team','matchid']).dropna() ## this contains our target variable
    ema_single_line = restructure_ema(ema_target,merge_vars,ema_vars)
    ema_results = pd.merge(single_line, ema_single_line, on=merge_vars,how="inner")
    return ema_results