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