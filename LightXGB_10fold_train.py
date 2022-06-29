# Avoiding warning
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn
# _______________________________

# Essential Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# _____________________________

np.random.seed(seed=142)

# scikit-learn :
from sklearn.linear_model import LogisticRegression, SGDClassifier

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, \
    RandomForestClassifier, \
    AdaBoostClassifier, \
    GradientBoostingClassifier, \
    ExtraTreesClassifier

#from catboost import CatBoostClassifier, Pool, cv
import lightgbm as lgb

Names = ['XGB']
Classifiers = [

     
]

def runClassifiers(args):
    
    df = pd.DataFrame(y)
   # df.to_csv('Original_Labels_1056_python.csv')

    from sklearn.metrics import accuracy_score, \
        log_loss, \
        classification_report, \
        confusion_matrix, \
        roc_auc_score,\
        average_precision_score,\
        auc,\
        roc_curve, f1_score, recall_score, matthews_corrcoef, auc

    
    #cv = StratifiedKFold(n_splits=10, shuffle=False)
    for classifier, name in zip(Classifiers, Names):
        accuray = []
        auROC = []
        avePrecision = []
        F1_Score = []
        AUC = []
        MCC = []
        Recall = []
        LogLoss = []

        mean_TPR = 0.0
        mean_FPR = np.linspace(0, 1, 100)

        CM = np.array([
            [0, 0],
            [0, 0],
        ], dtype=int)

        print('{} is done.'.format(classifier.__class__.__name__))
        print(classifier.__class__.__name__+'\n\n')


        model = classifier
        #ProbabilityScore_threshold=51
        ProbabilityScore=1 #for probability
        counterL = 21 # for label prediction
        decision_function_Score=31 # decision_function
        #FPRScore=1
        for (train_index, test_index) in cv.split(X, y):

            X_train = X[train_index]
            X_test = X[test_index]

            y_train = y[train_index]
            y_test = y[test_index]

            #rum = RandomUnderSampler(ratio='auto')
            #smote_enn = SMOTEENN(enn=EditedNearestNeighbours(), k=6, ratio='auto', smote=SMOTE())
               #         k=5, ratio='auto', smote=SMOTE())
            #sm = SMOTE(ratio='auto')

            #x_train_res, y_train_res = rum.fit_sample(X_train, y_train)
            #x_train_res, y_train_res = smote_enn.fit_sample(X_train, y_train)
            #x_train_res, y_train_res = sm.fit_sample(X_train, y_train)
            

           # np.savetxt(str(FPRScore) + '.csv', np.asarray(arr.round(3)))
           # FPRScore = FPRScore + 1

            FPR, TPR, threshold = roc_curve(y_test, ProbabilityScore)
            mean_TPR += np.interp(mean_FPR, FPR, TPR)
            mean_TPR[0] = 0.0
            #roc_auc = auc(FPR, TPR)
            mean_auc = auc(FPR, TPR)

            '''y_probaR = ProbabilityScore.round(3)
            np.savetxt(str(yProCounter) + '-Prob' + '.csv', np.asarray(y_probaR.round(3)))
            ProbabilityScore = ProbabilityScore + 1
            np.savetxt(str(yThreshCounter)+'-Thresh' + '.csv', np.asarray(y_proba_threshold.round(3)))
            y_proba_threshold = y_proba_threshold + 1
            np.savetxt(str(yDecisionfunCounter) + '-Decision-Func' + '.csv', np.asarray(y_pred_dec_func.round(3)))
            yDecisionfunCounter = yDecisionfunCounter + 1

            yProCounter+=1
            yThreshCounter+=1
            yDecisionfunCounter += 1'''

            round(3)), delimiter='  ', fmt='%f')

        #arr1 = np.asarray(TPR)
        #np.savetxt('TPR_Score.txt', np.asarray(arr1.round(3)), delimiter='  ', fmt='%f')

        plt.plot(
            mean_FPR,
            mean_TPR,
            linestyle='-',
            label='{} ({:0.3f})'.format(name, mean_auc), lw=2.0)

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Random')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=12)
        plt.ylabel('True Positive Rate (TPR)', fontsize=12)
        plt.title('ROC curve', fontweight='bold')
        plt.legend(loc='lower right')
        plt.savefig('ROC_Sg3_CS_PSSM_1056_Org1.png', dpi=300)
        plt.show()


if __name__ == '__main__':
    # print('Please, enter number of cross validation:')
    import argparse
   

    runClassifiers(args)