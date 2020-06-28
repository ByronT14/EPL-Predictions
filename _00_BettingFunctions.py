import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

def full_gambling_calc(df,stake,bookie_home_odds,bookie_draw_odds,bookie_away_odds,
bookie_home_prob,bookie_draw_prob,bookie_away_prob,model_home,model_draw,model_away):
    gambling_df = df.copy()
    gambling_df["Stake"] = stake
    gambling_df["HomeReturns"]=gambling_df[bookie_home_odds].apply(lambda x: returns(stake,x))
    gambling_df["DrawReturns"]=gambling_df[bookie_draw_odds].apply(lambda x: returns(stake,x))
    gambling_df["AwayReturns"]=gambling_df[bookie_away_odds].apply(lambda x: returns(stake,x))

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


def full_gambling_calc_cutoff(df,stake,bookie_home_odds,bookie_draw_odds,bookie_away_odds,
bookie_home_prob,bookie_draw_prob,bookie_away_prob,model_home,model_draw,model_away,index_cutoff):
    gambling_df = df.copy()
    gambling_df["Stake"] = stake
    gambling_df["HomeReturns"]=gambling_df[bookie_home_odds].apply(lambda x: returns(stake,x))
    gambling_df["DrawReturns"]=gambling_df[bookie_draw_odds].apply(lambda x: returns(stake,x))
    gambling_df["AwayReturns"]=gambling_df[bookie_away_odds].apply(lambda x: returns(stake,x))

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