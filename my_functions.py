
from unittest import result
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn.feature_selection import *
import plotly.express as px
import plotly.graph_objects as go
import pickle

def corr_coeff_2col(df, x, y):
    """
    Funzione che calcola il coefficiente di correlazione tra due colonne dello stesso dataframe
    Parametri:
    ----------\n
    df: Dataframe di partenza
    x: prima colonna
    y: seconda colonna\n
    Retern:
    -------\n
    Coefficienze di correlazione
    """
    x_mean = df[x].mean()
    y_mean = df[y].mean()
    # Calcolo la deviazione standard
    x_std = df[x].std()
    y_std = df[y].std()
    # Standardizzazione
    total_prod = (((df[x] - x_mean) / x_std) *
                      (((df[y] - y_mean) / y_std))).sum()
    corr = total_prod / (df.shape[0] - 1)
    return corr

def plot(df,Xcol,ycol1,ycol2,color1,color2,legend,xlabel,ylabel):
    ax = sns.lineplot(data=df,
                  x=Xcol,
                  y=ycol1,
                  palette=color1)
    sns.lineplot(data=df,
             x=Xcol,
             y=ycol2,
             palette=color2,
             ax=ax)
    ax.legend(legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
def find_nan_col(df_ls):
    outliers = []
    for df in df_ls:
        for k,v in df.isnull().sum().items():
            if float(v) > 0:
                outliers.append(k)
                return outliers

def common_items(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    if(set1 & set2):
        return list(set1 & set2)
    
def uncommon_items(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    if(set1 ^ set2):
        return list(set1 ^ set2) 
    else:
        return list()

def performance_calculator(models,X_tst,y_true,name_model:str):
    # Uso i modelli allenati per effettuare delle predizioni
    prediction1 = models.predict(X_tst)
    
    # Calcolo la probabilità delle predizioni
    prediction1_proba = (models.predict_proba(X_tst))[:, 1]
    
    # Calcolo precisione e MAE delle predizioni
    acc_1 = accuracy_score(y_true, prediction1).round(4)
    balanced_acc1 = balanced_accuracy_score(y_true, prediction1).round(4)
    precision_1 = precision_score(y_true, prediction1).round(4)
    recall_1 = recall_score(y_true, prediction1).round(4)
    f1_1 = f1_score(y_true, prediction1).round(4)
    mae_1 = mean_absolute_error(y_true, prediction1).round(4)

    # Printo a schermo le prestazioni medie
    print(f"Performance {name_model}:\n",
          f"Accuracy: {acc_1}\n",
          f"Balanced Accuracy: {balanced_acc1}\n",
          f"Precision: {precision_1}\n",
          f"Recall: {recall_1}\n",
          f"f1: {f1_1}\n",
          f"MAE: {mae_1}\n\n")
    
    return prediction1,prediction1_proba,acc_1,balanced_acc1,precision_1,recall_1,f1_1,mae_1
# Need to fix
def confusion_matrix_plot(y_t,y_pre,ttl,ax):
    cm = confusion_matrix(y_t, y_pre)
    sns.heatmap(data=cm, annot=True, fmt="1",cmap="vlag",ax=ax)
    #plt.title(ttl)
    
    
def roc_curve_plot(y_true,y_predicted_proba,label:str):
    y_test_roc = y_true.values.flatten()
    fpr1, tps1, threshold1 = roc_curve(y_test_roc, y_predicted_proba)
    auc1 = auc(fpr1, tps1)
    plt.plot(fpr1,
         tps1,
         linestyle="-",
         label=f"{label} (auc = {auc1.round(3)})")
    return fpr1,tps1,auc1

def precision_recall_plot(y_true,y_predicted_proba,label:str):
    y_test_pr = y_true.values.flatten()
    precision1, recall1, threshold3 = precision_recall_curve(y_test_pr, y_predicted_proba)
    plt.plot(precision1,
         recall1,
         linestyle="-",
         label=f"{label}")

def models_evaluation(models:list,X_train,y_tr,X_test,y_t,ada_param:dict,rf_param:dict,knn_params:dict,suffix:int,load_trained_models=True,iter=25,cv=10,load_csv=True):
    np.random.seed(42)
    param_ls = [ada_param,rf_param,knn_params]
    models_list =  models
    X_tr = X_train.reindex(sorted(X_train.columns), axis=1)
    X_t = X_test.reindex(sorted(X_test.columns), axis=1)
    
    # I create a list with the models paired with their parameters
    combined = []
    for idx,model in enumerate(models_list):
        combined.append([model,param_ls[idx]])

    # Calcola il punteggio di ciascun modello con 10 cross validation
    score_bs = []
    for model in models_list:
        name = str(model).split("(")[0]
        model_name = "file/" + name + f"_{suffix}_" + "csv_results" + ".pkl"
        
        if load_csv == False:
            score = cross_val_score(estimator=model,X=X_tr,y=y_tr,scoring="accuracy",cv=cv,n_jobs=-1)
            with open(model_name, "wb") as f:
                pickle.dump(score, f)
        
        elif load_csv == True:
            with open(model_name, "rb") as f:
                score = pickle.load(f)
            
        score_bs.append(score)
        print(f"Cross val score completed for {str(model)}\n")
    
    # Creo un dataframe per salvare i risultati
    score_bl_df = pd.DataFrame(data=score_bs, index=["Ada", "Rf", "Knn"])
    score_bl_df = score_bl_df.T
    print("Created dataframe with results\n")
    
    # Rimuove la colonna con la mediana dei punteggi più bassa (uso la mediana perchè è un valore robusto a differenza della media)
    a = []
    for col in score_bl_df.columns:
        min_col = score_bl_df.median().min()
        if score_bl_df[col].median() != min_col:
            a.append(col)
    if not "Ada" in a:
        combined.pop(0)
    elif not "Rf" in a:
        combined.pop(1)
    elif not "Knn" in a:
        combined.pop(2)
    print(f"The model with the lowest median of results was removed.\nThe remaining models are: {a}")
    
    # Uso RandomSearchCV per ottenere i migliori stimatori per ciascun modello
    fitted_models = []
    for model,param in combined:
        name = str(model).split("(")[0]
        model_name = "file/" + name + f"_{suffix}_" + ".pkl"
        
        # Esporto il modello allenato
        if load_trained_models == False:
            print(f"The {str(model)} model will now be trained on the provided trainset.")
            # Effettuo random search dei parametri passati
            rs = RandomizedSearchCV(estimator=model,cv=cv,param_distributions=param,n_iter=iter,n_jobs=-1,return_train_score=True,scoring="accuracy",random_state=42)
            rs_fitted = rs.fit(X_tr, y_tr)
            try:
                with open(model_name, "wb") as f:
                    pickle.dump(rs_fitted, f)
                fitted_models.append(rs_fitted)
                print(f"Random search of {model} was successful.\n")
            except Exception as e1:
                print(f"Errore nell'esportazione del modello: {model}\n",e1)
        
        # Importo i modelli allenati
        else:
            print(f"The {str(model)} model will now be imported.")
            try:
                with open(model_name, "rb") as f:
                    rs_fitted = pickle.load(f)
                fitted_models.append(rs_fitted)
                print(f"The {model} model was successfully imported.\n")
            
            except Exception as e2:
                print(f"Errore nell'importazione del modello: {model}\n",e2)
                
    # Prestazione dei modelli
    prediction1,prediction1_proba,acc_1,balanced_acc1,precision_1,recall_1,f1_1,mae_1 = performance_calculator(fitted_models[0],X_t,y_t,a[0])
    prediction2,prediction2_proba,acc_2,balanced_acc2,precision_2,recall_2,f1_2,mae_2 = performance_calculator(fitted_models[1],X_t,y_t,a[1])
    
    # Creo le confusion matrix per entrembi i modelli
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,9))
    for idx,cls in enumerate(zip(fitted_models, axes.flatten())):
        ConfusionMatrixDisplay.from_estimator(cls[0], 
                          X_t, 
                          y_t, 
                          ax=cls[1],
                          values_format=".0f", 
                          cmap='Blues')
        cls[1].title.set_text(a[idx])
    plt.tight_layout()  
    plt.show()
    
    # Creo curva ROC per i modelli
    names = ["fpr","tps","auc"]
    roc_results = []
    plt.figure(figsize=(16,9))
    for idx,proba in enumerate([prediction1_proba,prediction2_proba]):
        fpr = (names[0]+str(idx+1))
        tps = (names[1]+str(idx+1))
        auc = (names[2]+str(idx+1))
        fpr,fpr,auc = roc_curve_plot(y_t,proba,label=a[idx])
        roc_results.append([fpr,tps,auc])
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.show()
    
    
    # Precision Recall Curve
    plt.figure(figsize=(16,9))
    for idx,proba in enumerate([prediction1_proba,prediction2_proba]):
        precision_recall_plot(y_t,proba,label=a[idx])

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision - Recall curve")
    plt.legend()
    plt.show()
    
    return [[fitted_models[0].cv_results_,fitted_models[0].best_estimator_,roc_results[0][0],roc_results[0][1],roc_results[0][2],acc_1,balanced_acc1,mae_1,precision_1,recall_1,f1_1],
            [fitted_models[1].cv_results_,fitted_models[1].best_estimator_,roc_results[1][0],roc_results[1][1],roc_results[1][2],acc_2,balanced_acc2,mae_2,precision_2,recall_2,f1_2]]