from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn import metrics, model_selection as ms
from scipy.stats import poisson,skellam
import pandas as pd
import numpy as np

# def create_teamstr_df(df,span):
#     ema_output = ema_no_reset(df, span=span,feature_cols=features)
#     ema_target = pd.merge(ema_output, target_goals, on = ['f_Team','matchid']).dropna()
#     ema_target = pd.merge(ema_target,tm[["FFScout","Season","AMVIndex"]],left_on=["f_Team","season"],
#              right_on=["FFScout","Season"]).rename(columns={"AMVIndex":"f_AMVIndex"}).drop(["FFScout","Season"],axis=1)
#     ema_target = pd.merge(ema_target,final_order, on = ["season","gw_no"]).copy()
#     ema_target["opponent"] = ema_target.apply(lambda row: id_opponent(row["fixture"],row["f_HmGameFut"]),axis=1)
#     return ema_target


def fitStrModel(df,model, target, model_vars,test_df,cv):
    y = df[target]
    X = df[model_vars]
    scaled_glm = Pipeline([
        ('std', StandardScaler()),
        ('reg', model)
    ])
    GSMod = ms.GridSearchCV(estimator=scaled_glm,
                 param_grid={'reg__alpha': np.logspace(-10, 10, 21)},
                 scoring= ['neg_mean_squared_error'],
                 refit = 'neg_mean_squared_error',
                 return_train_score=True,
                 cv=cv)
    GSMod.fit(X,y)
    print("Target var is {} and the best train score is {}".format(target,GSMod.best_score_))
    test_score = metrics.mean_squared_error(test_df[target], GSMod.predict(test_df[model_vars]))
    print("The best score on the test dataset is {}".format(test_score))
    return GSMod


def fitScoreModel(df,model,target,model_vars,test_df,cv):
    X = df[model_vars]
    y = df[target]
    #glm = lm.LinearRegression()
    gsPredGoals = ms.GridSearchCV(estimator=model,
                     param_grid={'fit_intercept': [True]},
                     scoring= ['neg_mean_squared_error'],
                     refit = 'neg_mean_squared_error',
                     return_train_score=True,
                     cv=cv)
    gsPredGoals.fit(X,y)
    test_score = metrics.mean_squared_error(test_df[target], gsPredGoals.predict(test_df[model_vars]))
    print("Target var is {} and the best score is {}".format(target,gsPredGoals.best_score_))
    print("The best score on the test dataset is {}".format(test_score))
    #print(gsPredGoals.best_score_)
    #print(gsPredGoals.best_estimator_.coef_)
    return gsPredGoals

def simulate_match(home_goals_avg, away_goals_avg, max_goals=10):
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
    score_matrix = (np.outer(np.array(team_pred[0]), np.array(team_pred[1])))
    HomeProb = np.sum(np.tril(score_matrix, -1))
    DrawProb = np.sum(np.diag(score_matrix))
    AwayProb = np.sum(np.triu(score_matrix,1))
    #[HomeProb,DrawProb,AwayProb]
    return pd.Series([HomeProb,DrawProb,AwayProb])