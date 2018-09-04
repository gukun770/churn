import numpy as np

def standard_confusion_matrix(y_true,y_predict):
    tp = np.sum((y_predict == 1) & (y_predict == y_true))
    fp = np.sum((y_predict == 1) & (y_true == 0))
    fn = np.sum((y_predict == 0) & (y_true == 1))
    tn = np.sum((y_predict == 0) & (y_true == y_predict))
    confusion_matrix = np.array([[tp,fn],[fp,tn]])
    return confusion_matrix,fp,tp
    # """Make confusion matrix with format:
    #               -----------
    #               | TP | FP |
    #               -----------
    #               | FN | TN |
    #               -----------
    # Parameters
    # ----------
    # y_true : ndarray - 1D
    # y_pred : ndarray - 1D
    # Returns
    # -------
    # ndarray - 2D
    # """
    # [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    # return np.array([[tp, fp], [fn, tn]])

def profit_curve(cost_benefit, predicted_probs, labels):
    '''
    INPUTS:
    cost_benefit: your cost-benefit matrix
    predicted_probs: predicted probability for each datapoint (between 0 and 1)
    labels: true labels for each data point (either 0 or 1)

    OUTPUTS:
    array of profits and their associated thresholds
    '''
    idx = np.argsort(predicted_probs)
    predicted_probs= predicted_probs[idx]
    #predicted_probs = np.insert(predicted_probs,-1,1)

    labels = labels[idx]
    pred_temp = np.zeros(len(labels))
    thresholds = predicted_probs
    thresholds = np.insert(predicted_probs,0,0)

    cost = []
    for thresh in thresholds:

        pred_temp = np.zeros(len(labels))
        pred_temp[predicted_probs > thresh] = 1
        pred_temp[predicted_probs <= thresh] = 0
        conf, fpr,tpr,= standard_confusion_matrix(np.array(labels),np.array(pred_temp))

        cost.append(np.sum((conf*cost_benefit))/len(labels))


    return (np.array([cost,thresholds]))

def plot_profit_curve(model, cost_benefit, X_train, X_test, y_train, y_test,ax):
    model = model
    model.fit(X_train,y_train)
    test_probs = model.predict_proba(X_test)
    profits = profit_curve(cost_benefit, test_probs[:,1], y_test.values)
    profits = list(reversed(profits[0,:]))
    p = np.linspace(0,len(profits)/8,len(profits))

    ax.plot(p,profits,label=model.__class__.__name__)
    ax.grid(alpha = .4,color = 'r',linestyle = ':')
    ax.set_xlabel('Percentage of Test instances (decreasing by score)')
    ax.set_ylabel('Profit')
    ax.set_title('Profit Curves')
    return model.predict(X_test),profits,p

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