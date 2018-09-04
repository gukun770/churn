#Function to create the ROC curve:
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def roc_curve(probabilities,classification):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: list, list, list

    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    '''
    Class_df = pd.DataFrame([probabilities,classification])
    Class_df = Class_df.T
    Class_df.columns = ['Prob','class']

    Class_df = Class_df.sort_values('Prob')

    ThreshHolds = []
    FPR_list = []
    TPR_list = []
    for p in Class_df['Prob']:
        thresh = p

        Class_df['Model_label'] = np.where(Class_df['Prob'] >= thresh,1,0)#Create model label

        TPR_numerator = len(Class_df[(Class_df['class'] == 1) & (Class_df['Model_label'] == 1)])
        TPR_denom = Class_df['class'].sum()
        TPR  = float(TPR_numerator)/TPR_denom

        FPR_numerator = len(Class_df[(Class_df['class'] == 0) & (Class_df['Model_label'] == 1)])
        FPR_denom = len(Class_df[(Class_df['class'] == 0)])
        FPR = (float(FPR_numerator)/FPR_denom)

        #append to the lists
        ThreshHolds.append(thresh)
        FPR_list.append(FPR)
        TPR_list.append(TPR)
    return TPR_list,FPR_list,ThreshHolds

from sklearn.metrics import roc_curve
def plot_roc(X,y,models,ax):
    X_train, X_test, y_train, y_test= train_test_split(X,y,shuffle = True)


    for i,model in enumerate(models):
        model = model
        model.fit(X_train,y_train)
        test_probs = model.predict_proba(X_test)
        FPR_list,TPR_list,ThreshHolds = roc_curve(y_test,test_probs[:,0])
        ax.plot(TPR_list,FPR_list,
                 linestyle = '--',
                 linewidth = 2,
                 label = model.__class__.__name__)
    ax.plot([0,1],[0,1],ls = '--',c = 'navy',lw = 3)
    ax.grid(alpha = .3,ls = '-',c= 'g')
    ax.set_title('ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    return ax
