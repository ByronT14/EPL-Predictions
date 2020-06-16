def create_model_base(df,avg_method, span,ema_features, target_goals
                    ,merge_vars, calc_vars,single_line):
    import pandas as pd
    print(ema_features)
    ema_vars = ema_features + calc_vars
    ema_output = avg_method(df, span,ema_features)
    ema_target = pd.merge(ema_output, target_goals, on = ['f_Team','matchid']).dropna() ## this contains our target variable
    ema_single_line = restructure_ema(ema_target,merge_vars,ema_vars)
    ema_results = pd.merge(single_line, ema_single_line, on=merge_vars,how="inner")
    return ema_results


    def bet_profit(bet_correct, returns, bet_placed, stake):
    if bet_placed == 1:
        if bet_correct == 1:
            return returns
        else:
            return stake * -1
    else:
        return 0