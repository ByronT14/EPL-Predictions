import pandas as pd
from sklearn.preprocessing import LabelEncoder

#exponentially weighted averages - however this one won't reset after each season
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

def optimise_alpha(df, features,target, mod):
    """
    Function to get the cross validation scores of the target goals variable you're predicting
    df: dataframe containing features and target variable
    features: features to be used in the model
    target: Name of the target variable in your model dataset
    model: type of model to be run
    
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    df = df.dropna()
    X = df[features]
    y = df[target]
    y = y.apply(lambda x: float(x))
    model = mod
    kfold = StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
    ave_cv_score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kfold).mean()
    return ave_cv_score

def optimise_alpha_classifier(df, features,target, mod,metric):
    """
    Function to get the cross validation scores of the target goals variable you're predicting
    df: dataframe containing features and target variable
    features: features to be used in the model
    target: Name of the target variable in your model dataset
    model: type of model to be run
    
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    df = df.dropna()
    X = df[features]
    y = df[target]
    le = LabelEncoder()
    y = le.fit_transform(y)
    model = mod
    kfold = StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
    ave_cv_score = cross_val_score(model, X, y, scoring=metric, cv=kfold).mean()
    return ave_cv_score

def track_best_span(input,av_method,ema_features,classifier, model_vars):
    cv_scores = []
    lowest_span = 2
    for span in range(2, 46, 2):
        model_base = create_model_base(input,av_method,span,ema_features)
        model_base = model_base.dropna()
        error = optimise_alpha_classifier(model_base,model_vars,"Result",classifier,"neg_log_loss")
        cv_scores.append(error)
        if error == max(cv_scores):
            lowest_span = span
    return cv_scores

def create_model_probs_df(df,seasons,classifier,input_vars,classes,keep_vars):
    base = df[df['season'].isin(seasons)].copy()
    predictions = pd.DataFrame(classifier.predict_proba(base[input_vars]),columns=classes)
    pred_cols = ["Model_Prob_"+col for col in predictions.columns]
    predictions = predictions.rename(columns={col:'Model_Prob_'+col for col in predictions.columns})
    base = pd.concat([base,predictions],axis=1)
    #base = base[merge_vars+team_vars+pred_cols]
    base = base[keep_vars]
    return base

def returns(stake, odds):
    returns = stake * odds - stake
    return returns


def expected_returns(stake, returns, model_prob):
    p = model_prob
    exp_returns = (returns * p) - ((1 - p) * stake)
    return exp_returns


def result_returns(row):
    # if np.isnan(row["PredResult"])==False:
    if row["Result"] == "H":
        return row["HomeReturns"]
    if row["Result"] == "D":
        return row["DrawReturns"]
    if row["Result"] == "A":
        return row["AwayReturns"]


def determine_bet(row):
    # if np.isnan(row["PredResult"])==False:
    if row["BestValueExpReturns"] == row["ExpHomeReturns"]:
        return "H"
    if row["BestValueExpReturns"] == row["ExpDrawReturns"]:
        return "D"
    if row["BestValueExpReturns"] == row["ExpAwayReturns"]:
        return "A"


def most_likely_result(home_prob, draw_prob, away_prob):
    most_likely = max(home_prob, draw_prob, away_prob)
    if most_likely == home_prob:
        return "H"
    if most_likely == draw_prob:
        return "D"
    if most_likely == away_prob:
        return "A"


def prediction_correct(result, prediction):
    if result == prediction:
        return 1
    else:
        return 0


def bet_index_cutoff(cutoff, bestindex):
    if bestindex >= cutoff:
        return 1
    else:
        return 0

def full_gambling_calc(df,stake,bookie_home_prob,bookie_draw_prob,bookie_away_prob,model_home,model_draw,model_away):
    gambling_df = df.copy()
    gambling_df["Stake"] = stake
    gambling_df["HomeReturns"]=gambling_df["BookieHomeOdds"].apply(lambda x: returns(stake,x))
    gambling_df["DrawReturns"]=gambling_df["BookieDrawOdds"].apply(lambda x: returns(stake,x))
    gambling_df["AwayReturns"]=gambling_df["BookieAwayOdds"].apply(lambda x: returns(stake,x))

    gambling_df["ResultReturns"]= gambling_df.apply(lambda row: result_returns(row), axis =1)

    gambling_df["ExpHomeReturns"] = gambling_df.apply(lambda row: 
                                                      expected_returns(row["Stake"],row["HomeReturns"],row[model_home])
                                                    , axis = 1)
    gambling_df["ExpDrawReturns"] = gambling_df.apply(lambda row: expected_returns(row.Stake,row.DrawReturns,row[model_draw])
                                                    , axis = 1)
    gambling_df["ExpAwayReturns"] = gambling_df.apply(lambda row: expected_returns(row.Stake,row.AwayReturns,row[model_away])
                                                    , axis = 1)

    gambling_df["BetIndexHome"] = gambling_df[model_home]/gambling_df[bookie_home_prob]
    gambling_df["BetIndexDraw"] = gambling_df[model_draw]/gambling_df[bookie_draw_prob]
    gambling_df["BetIndexAway"] = gambling_df[model_away]/gambling_df[bookie_away_prob]

    gambling_df["BestValueExpReturns"]= gambling_df[["ExpHomeReturns","ExpDrawReturns","ExpAwayReturns"]].max(axis=1)
    gambling_df["BestValueIndex"]=gambling_df[["BetIndexHome","BetIndexDraw","BetIndexAway"]].max(axis=1)
    
    gambling_df["BookiePrediction"]=gambling_df.apply(lambda row: most_likely_result(row[bookie_home_prob],row[bookie_draw_prob],
                                                                                    row[bookie_away_prob]),axis=1)
    gambling_df["ModelPrediction"]=gambling_df.apply(lambda row: most_likely_result(row[model_home],row[model_draw]
                                                                                    ,row[model_away]),axis =1)
    gambling_df["ValuePrediction"] = gambling_df.apply(lambda row: determine_bet(row),axis = 1)
    
    gambling_df["BookiePredCorrect"] = gambling_df.apply(lambda row: prediction_correct(row["Result"],row["BookiePrediction"]),axis=1)
    gambling_df["ModelPredCorrect"]= gambling_df.apply(lambda row:prediction_correct(row["Result"],row["ModelPrediction"]),axis=1)
    gambling_df["ValuePredCorrect"]= gambling_df.apply(lambda row:prediction_correct(row["Result"],row["ValuePrediction"]),axis=1)
    
    return gambling_df

def bet_profit(bet_correct, returns, bet_placed, stake):
    if bet_placed == 1:
        if bet_correct == 1:
            return returns
        else:
            return stake * -1
    else:
        return 0


def full_gambling_calc_cutoff(df,stake,bookie_home_prob,bookie_draw_prob,bookie_away_prob,model_home,model_draw,model_away,index_cutoff):
    gambling_df = df.copy()
    gambling_df["Stake"] = stake
    gambling_df["HomeReturns"]=gambling_df["BookieHomeOdds"].apply(lambda x: returns(stake,x))
    gambling_df["DrawReturns"]=gambling_df["BookieDrawOdds"].apply(lambda x: returns(stake,x))
    gambling_df["AwayReturns"]=gambling_df["BookieAwayOdds"].apply(lambda x: returns(stake,x))

    gambling_df["ResultReturns"]= gambling_df.apply(lambda row: result_returns(row), axis =1)

    gambling_df["ExpHomeReturns"] = gambling_df.apply(lambda row: 
                                                      expected_returns(row["Stake"],row["HomeReturns"],row[model_home])
                                                    , axis = 1)
    gambling_df["ExpDrawReturns"] = gambling_df.apply(lambda row: expected_returns(row.Stake,row.DrawReturns,row[model_draw])
                                                    , axis = 1)
    gambling_df["ExpAwayReturns"] = gambling_df.apply(lambda row: expected_returns(row.Stake,row.AwayReturns,row[model_away])
                                                    , axis = 1)

    gambling_df["BetIndexHome"] = gambling_df[model_home]/gambling_df[bookie_home_prob]
    gambling_df["BetIndexDraw"] = gambling_df[model_draw]/gambling_df[bookie_draw_prob]
    gambling_df["BetIndexAway"] = gambling_df[model_away]/gambling_df[bookie_away_prob]

    gambling_df["BestValueExpReturns"]= gambling_df[["ExpHomeReturns","ExpDrawReturns","ExpAwayReturns"]].max(axis=1)
    gambling_df["BestValueIndex"]=gambling_df[["BetIndexHome","BetIndexDraw","BetIndexAway"]].max(axis=1)
    
    gambling_df["BookiePrediction"]=gambling_df.apply(lambda row: most_likely_result(row[bookie_home_prob],row[bookie_draw_prob],
                                                                                    row[bookie_away_prob]),axis=1)
    gambling_df["ModelPrediction"]=gambling_df.apply(lambda row: most_likely_result(row[model_home],row[model_draw]
                                                                                    ,row[model_away]),axis =1)
    gambling_df["ValuePrediction"] = gambling_df.apply(lambda row: determine_bet(row),axis = 1)
    
    gambling_df["BookiePredCorrect"] = gambling_df.apply(lambda row: prediction_correct(row["Result"],row["BookiePrediction"]),axis=1)
    gambling_df["ModelPredCorrect"]= gambling_df.apply(lambda row:prediction_correct(row["Result"],row["ModelPrediction"]),axis=1)
    gambling_df["ValuePredCorrect"]= gambling_df.apply(lambda row:prediction_correct(row["Result"],row["ValuePrediction"]),axis=1)
    gambling_df["ValueRequirementMet"] = gambling_df.apply(lambda row: bet_index_cutoff(index_cutoff,row["BestValueIndex"]),axis=1)
    gambling_df["ModelPredProfit"] = gambling_df.apply(lambda row: 
                                                            bet_profit(row["ModelPredCorrect"],row["ResultReturns"],1,row["Stake"])
                                                            ,axis=1)
    gambling_df["ValuePredProfit"] = gambling_df.apply(lambda row: 
                                                            bet_profit(row["ValuePredCorrect"],row["ResultReturns"],1,row["Stake"])
                                                            ,axis=1)
    gambling_df["ValueCutoffProfit"] = gambling_df.apply(lambda row: 
                                                            bet_profit(row["ValuePredCorrect"],row["ResultReturns"]
                                                                        ,row["ValueRequirementMet"],row["Stake"])
                                                            ,axis=1)
    
    return gambling_df