import numpy as np
import pandas as pd

def blank_mat():
    '''
    Creates a dataframe that's blank
    '''
    matrix = np.zeros((11,11),dtype = int)
    df_blank = pd.DataFrame(matrix)
    return df_blank

def home_mat():
    '''
    Creats a Datframe of 1s for all home victories
    '''
    matrix = np.tril(np.ones((11, 11), dtype=int), -1)
    return pd.DataFrame(matrix)

def away_mat():
    '''
    Creats a Datframe of 1s for all away victories
    '''
    matrix = np.triu(np.ones((11, 11), dtype=int), 1)
    return pd.DataFrame(matrix)

def draw_mat():
    '''
    Creats a Datframe of 1s for all away victories
    '''
    matrix = np.zeros((11, 11), dtype=int)
    np.fill_diagonal(matrix,1)
    return pd.DataFrame(matrix)

def wrong_mat():
    '''
    Creates a dataframe that's blank
    '''
    matrix = np.ones((11,11),dtype = int)
    return pd.DataFrame(matrix)

def goal_diff_points():
    matrix = np.ones((11,11),dtype = int)*15
    np.fill_diagonal(matrix,20)
    return pd.DataFrame(matrix)

def pred_score(score_matrix):
    max_prob = score_matrix.values.max()
    min_prob = score_matrix.values.min()
    pred_hg = np.where(score_matrix.values == max_prob)[0][0]
    pred_ag = np.where(score_matrix.values == max_prob)[1][0]
    pred_score  = str(pred_hg) + ' . ' + str(pred_ag)
    pred_score_prob = max_prob
    return pd.Series([pred_score,pred_score_prob])

def ikts_points(score_matrix,home_df,draw_df,away_df,wrong_df,gd_points_mat):
    df_mat = score_matrix
    mat_gd = pd.DataFrame(np.zeros((11,11),dtype = int))
    for i in range(0,11):
        for j in range(0,11):
            prob_gd = 0
            for k in range(0,11):
                for l in range(0,11):
                    if i-j == k-l:
                        prob_gd = prob_gd + df_mat.loc[k,l]
            #print(prob_gd)
            mat_gd.loc[i,j] = prob_gd
    # Home probability matrix       
    home_prob_mat = ((home_df*df_mat).values.sum())*home_df
    home_prob = (home_df*df_mat).values.sum()
    
    # Away probability matrix
    away_prob_mat = ((away_df*df_mat).values.sum())*away_df
    away_prob = (away_df*df_mat).values.sum()
    
    #draw probability matrix
    draw_prob_mat = ((draw_df*df_mat).values.sum())*draw_df
    draw_prob = (draw_df*df_mat).values.sum()
    
    #probability matrix that the it will be the right goal difference but not the right score
    ikts_gd_prob = mat_gd - df_mat
    
    #probability matrix that it will be the right outcome but not the right goal difference
    ikts_outcome_prob = (home_prob_mat+draw_prob_mat+away_prob_mat) - mat_gd
    
    #probability matrix that it will be the wrong outcome (1 minus result probability)
    wrong_prob = wrong_df - (home_prob_mat+draw_prob_mat + away_prob_mat)
    
    #Matrix of expected IKTS points - correct score prob*30 or correct goal difference or correct outcome or incorrect
    ikts_points = (df_mat*30) + (ikts_gd_prob*gd_points_mat) + (ikts_outcome_prob*10) + (wrong_prob*-10)
    return ikts_points

def IKTSPoints(pred_score,homegoals, awaygoals):
    pred_home = int(pred_score.split(' . ')[0])
    pred_away = int(pred_score.split(' . ')[1])
    if (pred_home == homegoals) & (pred_away == awaygoals):
        score = 30
    elif (pred_home == pred_away) & (homegoals == awaygoals):
        score = 20
    elif (pred_home-pred_away) == (homegoals - awaygoals):
        score = 15
    elif (pred_home-pred_away>0) & (homegoals - awaygoals>0):
        score = 10
    elif (pred_home-pred_away <0) & (homegoals - awaygoals<0):
        score = 10
    else:
        score = -10
    return score

def chip_placement(xp,insurance,banker):
    if xp == insurance:
        return "Insurance"
    elif xp == banker:
        return "Banker"
    else:
        return ""