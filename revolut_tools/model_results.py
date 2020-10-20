import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def scores(model, X_train, X_val, y_train, y_val):
    """
    Get results of a fitted model comparing train and test

    Args:
        X_train: train features
        X_val: test features
        y_train: train target
        y_val: test target

    Returns:
        None
    """
    train_prob = model.predict_proba(X_train)[:,1]
    val_prob = model.predict_proba(X_val)[:,1]
    train = roc_auc_score(y_train,train_prob)
    val = roc_auc_score(y_val,val_prob)
    print('train:',round(train,2),'test:',round(val,2))

def opt_plots(opt_model):
    """
    Plot a grid plot of the different results of a gridsearch params.

    Args:
        opt_model: gridsearch fitted model

    Returns:
        None
    """
    opt = pd.DataFrame(opt_model.cv_results_)
    cols = [col for col in opt.columns if ('mean' in col or 'std' in col) and 'time' not in col]
    params = pd.DataFrame(list(opt.params))
    opt = pd.concat([params,opt[cols]],axis=1,sort=False)

    parameter_1 = list(opt_model.param_grid.keys())[0]
    parameter_2 = list(opt_model.param_grid.keys())[1]

    plt.figure(figsize=[15,4])
    plt.subplot(121)
    sns.heatmap(
        pd.pivot_table(
            opt,index=parameter_1, columns=parameter_2, values='mean_train_score'
        )*100
    )
    plt.title('ROC_AUC - Training')
    plt.subplot(122)
    sns.heatmap(
        pd.pivot_table(
            opt,index=parameter_1,columns=parameter_2,values='mean_test_score'
        )*100
    )
    plt.title('ROC_AUC - Validation')
