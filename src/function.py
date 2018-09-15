import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
plt.style.use('ggplot')

def profit_curve(cost_benefit, predicted_probs, labels):
    """Function to calculate list of profits based on supplied cost-benefit
    matrix and prediced probabilities of data points and thier true labels.
    Parameters
    ----------
    cost_benefit    : ndarray - 2D, with profit values corresponding to:
                                          -----------
                                          | TP | FP |
                                          -----------
                                          | FN | TN |
                                          -----------
    predicted_probs : ndarray - 1D, predicted probability for each datapoint
                                    in labels, in range [0, 1]
    labels          : ndarray - 1D, true label of datapoints, 0 or 1
    Returns
    -------
    profits    : ndarray - 1D
    thresholds : ndarray - 1D
    """
    n_obs = float(len(labels))
    # Make sure that 1 is going to be one of our thresholds
    maybe_one = [] if 1 in predicted_probs else [1] 
    thresholds = maybe_one + sorted(predicted_probs, reverse=True)
    profits = []
    for threshold in thresholds:
        y_predict = predicted_probs >= threshold
        confusion_matrix = standard_confusion_matrix(labels, y_predict)
        threshold_profit = np.sum(confusion_matrix * cost_benefit) / n_obs
        profits.append(threshold_profit)
    return np.array(profits), np.array(thresholds)

def standard_confusion_matrix(y_true, y_pred):
    """Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Parameters
    ----------
    y_true : ndarray - 1D
    y_pred : ndarray - 1D
    Returns
    -------
    ndarray - 2D
    """
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    return np.array([[tp, fp], [fn, tn]])

def get_model_profits(model, cost_benefit, X_test, y_test):
    """Predicts passed model on testing data and calculates profit from cost-benefit
    matrix at each probability threshold.
    Parameters
    ----------
    model           : sklearn model - need to implement predict
    cost_benefit    : ndarray - 2D, with profit values corresponding to:
                                          -----------
                                          | TP | FP |
                                          -----------
                                          | FN | TN |
                                          -----------
    X_test          : ndarray - 2D
    y_test          : ndarray - 1D
    Returns
    -------
    model_profits : model, profits, thresholds
    """
    predicted_probs = model.predict_proba(X_test)[:, 1]
    profits, thresholds = profit_curve(cost_benefit, predicted_probs, y_test)

    return profits, thresholds

def plot_model_profits(model_profits, save_path=None):
    """Plotting function to compare profit curves of different models.
    Parameters
    ----------
    model_profits : list((model, profits, thresholds))
    save_path     : str, file path to save the plot to. If provided plot will be
                         saved and not shown.
    """
    for model, profits, threshold in model_profits:
        # percentages = np.linspace(0, 100, profits.shape[0])
        plt.plot(threshold, profits, label=model.__class__.__name__)

    plt.title("Profit Curves")
    plt.xlabel("Threshold used to classifier churn")
    plt.ylabel("Profit")
    plt.legend(loc='best')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def find_best_threshold(model_profits):
    """Find model-threshold combo that yields highest profit.
    Parameters
    ----------
    model_profits : list((model, profits, thresholds))
    Returns
    -------
    max_model     : str
    max_threshold : float
    max_profit    : float
    """
    max_model = None
    max_threshold = None
    max_profit = None
    for model, profits, thresholds in model_profits:
        max_index = np.argmax(profits)
        if not max_model or profits[max_index] > max_profit:
            max_model = model
            max_threshold = thresholds[max_index]
            max_profit = profits[max_index]
    return max_model, max_threshold, max_profit

def plot_(df,target,ax):
    ax[0].hist(df.trips_in_first_30_days[target == 0],bins = list(np.linspace(0,20,50)),alpha = .6,label = 'no churn',normed = 1);
    ax[0].hist(df.trips_in_first_30_days[target == 1],bins = list(np.linspace(0,20,50)),alpha = .6,label = '30 day churn',normed = 1);
    ax[0].set_xlim([-1,20])
    ax[0].legend()
    ax[0].set_xlabel('Number of rides in the First 30 days')
    ax[0].set_ylabel('Normalized Count');
    ax[0].grid(alpha = .2,color = 'r',linestyle = '--')
    ax[0].set_title('Number of rides in First 30 days hist')
    ax[1].hist(df.weekday_pct[target == 0],alpha = .6,label = 'no churn',normed = 1);
    ax[1].hist(df.weekday_pct[target == 1],alpha = .6,label = '30 day churn',normed = 1);
    ax[1].set_xlim([-1,110])
    ax[1].legend()
    ax[1].set_xlabel('Week day Percent')
    ax[1].set_title('Weekday Percent hist')
    ax[1].grid(alpha = .2,color = 'r',linestyle = '--')
    ax[2].hist(df.surge_pct[target == 0],alpha = .6,label = 'no churn',normed = 1);
    ax[2].hist(df.surge_pct[target == 1],alpha = .6,label = '30 day churn',normed = 1);
    ax[2].set_xlim([-1,110])
    ax[2].legend()
    ax[2].set_xlabel('Surge Percent')
    ax[2].set_title('Surge Percent hist')
    ax[2].grid(alpha = .2,color = 'r',linestyle = '--')
    ax[3].hist(df.avg_dist[target == 0],bins = list(np.linspace(0,60,40)),alpha = .6,label = 'no churn',normed = 1);
    ax[3].hist(df.avg_dist[target == 1],bins = list(np.linspace(0,60,40)),alpha = .6,label = '30 day churn',normed = 1);
    ax[3].set_xlim([-1,40])
    ax[3].legend()
    ax[3].set_xlabel('Avg Distance')
    ax[3].set_title('Average Distance hist')
    ax[3].grid(alpha = .2,color = 'r',linestyle = '--')
    return ax