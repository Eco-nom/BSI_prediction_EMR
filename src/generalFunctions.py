import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss

# Grid search
# globalVariables module 
from globalVariables import Feature_lst
from globalVariables import Feature_OHE

# clf=LogisticRegression(C=1, class_weight='balanced', max_iter=20000, random_state=seed)

# USAGE : X,Y = split_XY(df_merge)
# Input - df_merge (pandas dataframe)
# Output - X, Y (pandas dataframe)
def splitXY(df_merge, all_features):
    X = df_merge[all_features]
    Y = df_merge['Septic_shock2_visit']
    #Y = df_merge['BSI']
    #Y = df_merge['Septic_shock']

    return X, Y


# USAGE : X_train, X_test, y_train, y_test = splitTrainTest(X, Y, seed, testsize)

def splitTrainTest(X, Y, seed, testsize):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, random_state=seed, test_size=testsize)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print('Train Test split completed : ( Test size %.2f )' % testsize)

    return X_train, X_test, y_train, y_test


# USAGE : X_train, X_test = standardScaleData(X_train, X_test) # Recursive
# Standard scaling - Logistic regression 

def standardScaleData(X_train, X_test):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[Feature_lst])
    X_test_scaled = scaler.fit_transform(X_test[Feature_lst])
    scaled_data_X_train = np.concatenate([X_train_scaled, X_train[Feature_OHE]], axis=1)
    scaled_data_X_test = np.concatenate([X_test_scaled, X_test[Feature_OHE]], axis=1)

    return scaled_data_X_train, scaled_data_X_test

# USAGE : lst_selected_idx, lst_selected_featurename = selectFeatures(clf, number_of_features, X_train, y_train, all_features, X)

def selectFeatures(clf, number_of_features, X_train, y_train, all_features, X):
    rfe = RFE(clf, n_features_to_select=number_of_features)
    rfe = rfe.fit(X_train, y_train)

    idx = 0
    lst_selected_idx = []
    for i in rfe.support_:
        if i == True:
            lst_selected_idx.append(idx)
        idx += 1

    lst_selected_featurename = []

    for i in X.iloc[0, lst_selected_idx].index:
        lst_selected_featurename.append(i)

    return lst_selected_idx, lst_selected_featurename

# USAGE : lst_selected_idx, lst_selected_featurename = selectFeatures(clf, number_of_features, X_train, y_train, all_features, X)

def selectFeatures_XGBoost(clf, number_of_features, X_train, y_train, all_features, X):
    rfe = RFE(clf, n_features_to_select=number_of_features)
    rfe = rfe.fit(X_train, y_train)

    idx = 0
    lst_selected_idx = []
    for i in rfe.support_:
        if i == True:
            lst_selected_idx.append(idx)
        idx += 1

    lst_selected_featurename = X.columns[rfe.support_].tolist() # Selected feature names

    return lst_selected_idx, lst_selected_featurename


# Logistic regression 
# USAGE : coef_result = extractCoefficient(clf, X_train, y_train, lst_selected_featurename)
# Output format - pandas dataframe
def extractCoefficient(clf, X_train, y_train, lst_selected_featurename):
    clf.score(X_train, y_train)
    coef_result = pd.DataFrame({'Var': lst_selected_featurename, 'Coefficient': list(clf.coef_[0])})

    return coef_result
# ROCAUC
# pred_proba = clf.predict_proba(X_test)[:,1]
def plotROC(clf, X_test, y_test, pred_proba, modelname):
    logit_roc_auc = roc_auc_score(y_test, pred_proba)
    fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
    plt.figure(dpi=300)
    plt.plot(fpr, tpr, label='%s (area = %0.3f)' % (modelname, logit_roc_auc))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic') 
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
    plt.clf()



# USAGE : plotPRcurve(y_test, clf.predict_proba(X_test)[:,1])
def plotPRcurve(y_test, pred_proba_c1):
    from sklearn.metrics import precision_recall_curve
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_c1)

   
    # plt.figure(figsize=(8, 6))
    plt.figure(dpi=300)
    thresholds_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0: thresholds_boundary], linestyle='--', label='precision')
    plt.plot(thresholds, recalls[0: thresholds_boundary], label='recall')

    
    stard, end = plt.xlim()
    plt.xticks(np.round(np.arange(stard, end, 0.1), 2))

    
    plt.title('Precision-Recall curve')
    plt.xlabel('Threshold value')
    plt.ylabel('Precision and Recall value')
    plt.legend()
    plt.grid()
    plt.show()
    plt.clf()


##########################

def plotKfoldROC(X_train, y_train, clf, kfold, modelname):
    #plt.figure(dpi=300)
    fold = 1
    mean_auc = []
    mean_pr_auc=[] 
    for train_index, test_index in kfold.split(X_train, y_train):
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
        clf.fit(X_train_fold, y_train_fold)

        #clf_roc_auc = roc_auc_score(y_test_fold, clf.predict(X_test_fold))
        clf_roc_auc = roc_auc_score(y_test_fold, clf.predict_proba(X_test_fold)[:, 1])
        fpr, tpr, thresholds = roc_curve(y_test_fold, clf.predict_proba(X_test_fold)[:, 1])
        mean_auc.append(clf_roc_auc)
        #plt.plot(fpr, tpr, label='Fold %s (area = %0.3f)' % (fold, clf_roc_auc))
        #plt.plot([0, 1], [0, 1], 'r--')
        #plt.xlim([0.0, 1.0])
        #plt.ylim([0.0, 1.05])
        fold += 1
        #plt.xlabel('False Positive Rate')
        #plt.ylabel('True Positive Rate')
        #plt.title('Receiver operating characteristic\n%s' % modelname)
        #plt.legend(loc="lower right")
        #plt.savefig('Log_ROC')
        
        prec, recall, _ = precision_recall_curve(y_test_fold, clf.predict_proba(X_test_fold)[:, 1])
        auc_score = auc(recall, prec)
        mean_pr_auc.append(auc_score)
    cv_roc_auc = np.mean(mean_auc)
    cv_pr_auc=np.mean(mean_pr_auc)
    print('Mean ROC AUC score : %.3f' % cv_roc_auc)
    #plt.show()
    #plt.clf()
    return cv_roc_auc, cv_pr_auc


#########################################################################
# USAGE : pr_auc_plot(X_train_test,y_train_test,X_test_test,y_test_test)
def plotPRAUC(clf,X_train, y_train, X_test, y_test, seed, modelname):
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import auc
    from sklearn.dummy import DummyClassifier  
    from sklearn.metrics import PrecisionRecallDisplay
    
    model = DummyClassifier(strategy='stratified', random_state=seed) 
    model.fit(X_train, y_train)
    yhat_no = model.predict_proba(X_test)
    pos_probs_no = yhat_no[:, 1]
    # calculate the precision-recall auc
    precision_no, recall_no, _ = precision_recall_curve(y_test, pos_probs_no)
    noskill_auc_score = auc(recall_no, precision_no)
    no_skill = (len(y_test[y_test == 1]) + len(y_train[y_train == 1])) / (len(y_test) + len(y_train))
    # plot the no skill precision-recall curve
    plt.figure(dpi=300)
    
    y_pred = clf.predict_proba(X_test)[:, 1]  # pos_probs
    prec, recall, _ = precision_recall_curve(y_test, y_pred)
    disp = PrecisionRecallDisplay(precision=prec, recall=recall)
    auc_score = auc(recall, prec)
    
    disp.plot(marker='.', label='%s PR AUC = %.3f' % (modelname, auc_score))
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill PR AUC = %.3f' % noskill_auc_score)
    plt.title('%s PR AUC' % modelname)
    plt.legend()
    plt.show()
    plt.clf()
    print('No skill PR AUC : %.3f' % noskill_auc_score)
    print('%s PR AUC : %.3f' % (modelname, auc_score))
    # plt.plot(fpr, tpr, label='%s (area = %0.2f)' % (modelname,logit_roc_auc))


##############################################################################################################
def logloss_plot(clf, X_test):
    print('Log loss plot')
    yhat = []
    yhat = list(clf.predict_proba(X_test)[:, 1])
    yhat.sort()
    log_loss_0 = [log_loss([0], [x], labels=[0, 1]) for x in yhat]
    log_loss_1 = [log_loss([1], [x], labels=[0, 1]) for x in yhat]
    # plt.title='Log loss plot'
    plt.figure(dpi=300)
    plt.plot(yhat, log_loss_0, 'ro-', label='True=0')
    plt.plot(yhat, log_loss_1, 'bo-', label='True=1')
    plt.legend()
    plt.show()
    plt.clf()


def brierloss_plot(clf, X_test):
    print('Brier loss plot')
    yhat = []
    yhat = list(clf.predict_proba(X_test)[:, 1])
    yhat.sort()
    log_loss_0 = [brier_score_loss([0], [x], pos_label=1) for x in yhat]
    log_loss_1 = [brier_score_loss([1], [x], pos_label=1) for x in yhat]
    # plt.title='Brier score loss plot'
    plt.figure(dpi=300)
    plt.plot(yhat, log_loss_0, 'ro-', label='True=0')
    plt.plot(yhat, log_loss_1, 'bo-', label='True=1')
    plt.legend()
    plt.show()
    plt.clf()


##########################################################################
def plotCalibratedProb(clf,X_train, y_train, kfold, modelname):
    from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
    from sklearn.calibration import calibration_curve

    def cal_hist(yhat_calibrated):
        print('Fold %i' % fold)
        fig = plt.figure()
        # y_prob = clf.predict_proba(X_test)
        y_prob = yhat_calibrated

        ax = fig.add_subplot()

        ax.hist(
            # calibration_displays.y_prob,
            y_prob,
            range=(0, 1),
            bins=10,
            label='%s' % modelname,
            # color=colors(i),
        )
        ax.set(title='%s' % modelname, xlabel="Mean predicted probability", ylabel="Count")

    ##
    def calibrated(X_train, y_train, X_test):
        calibrated = CalibratedClassifierCV(clf, method="sigmoid", cv=5)
        calibrated.fit(X_train, y_train)
        return calibrated.predict_proba(X_test)[:, 1]

    #plt.figure(dpi=300)
    fold = 1
    briers_lst=[] 
    for train_index, test_index in kfold.split(X_train, y_train):
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
        clf.fit(X_train_fold, y_train_fold)
        yhat_calibrated = calibrated(X_train_fold, y_train_fold, X_test_fold)
        fop_calibrated, mpv_calibrated = calibration_curve(y_test_fold, yhat_calibrated, n_bins=10)
        brier = brier_score_loss(y_test_fold, yhat_calibrated, pos_label=1)
        briers_lst.append(brier) # 추가 20220729
        print(fold)
        print(clf)
        #plt.plot([0, 1], [0, 1], linestyle='--', color='black')
        #plt.plot(mpv_calibrated, fop_calibrated, marker='.', label='Fold %i (%.3f)' % (fold, brier))
        fold += 1
        #plt.xlabel('Mean predicted value (Positive class : 1)')
        #plt.ylabel('Fraction of positives (Positive class : 1)')
        #plt.title('%s\nCalibration plot' % modelname)
        #plt.legend(loc="lower right")
    #plt.show()
    #plt.clf()
    return np.mean(briers_lst) 

# USAGE : score_df = hyperParamOptimization(parameters, custom_scorer, X_train, y_train)
# from sklearn.metrics import fbeta_score, make_scorer
# parameters = {'C': [0.0001, 0.001, 0.01, 0.1]}
# custom_scorer = make_scorer(fbeta_score, beta=2)  # roc_auc , f1 , accuracy etc.
def hyperParamOptimization(clf,parameters, custom_scorer, X_train, y_train):
    from sklearn.model_selection import GridSearchCV
    grid_clf = GridSearchCV(clf, param_grid=parameters, cv=5, refit=True, scoring=custom_scorer, verbose = 1)
    grid_clf.fit(X_train, y_train)
    score_df = pd.DataFrame(grid_clf.cv_results_)
    return score_df


# 20220620

import numpy as np
from sklearn.metrics import roc_curve

def sensitivity_specifity_cutoff(y_true, y_score):
    '''Find data-driven cut-off for classification
    
    Cut-off is determied using Youden's index defined as sensitivity + specificity - 1.
    
    Parameters
    ----------
    
    y_true : array, shape = [n_samples]
        True binary labels.
        
    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        “decision_function” on some classifiers).
        
    References
    ----------
    
    Ewald, B. (2006). Post hoc choice of cut points introduced bias to diagnostic research.
    Journal of clinical epidemiology, 59(8), 798-801.
    
    Steyerberg, E.W., Van Calster, B., & Pencina, M.J. (2011). Performance measures for
    prediction models and markers: evaluation of predictions and classifications.
    Revista Espanola de Cardiologia (English Edition), 64(9), 788-794.
    
    Jiménez-Valverde, A., & Lobo, J.M. (2007). Threshold criteria for conversion of probability
    of species presence to either–or presence–absence. Acta oecologica, 31(3), 361-369.
    '''
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]
