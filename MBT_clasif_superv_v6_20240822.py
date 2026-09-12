# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:04:30 2023

@author: yasma
"""

#====================
#dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb

from sklearn.model_selection import train_test_split

# import PCA 
from sklearn.decomposition import PCA
# import LDA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# escalar los datos
from sklearn.preprocessing import MinMaxScaler

#clasificadores
from sklearn import svm
from sklearn import tree
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

import pickle

# #===========================================================
# #===========================================================

# #leer datos
# path1='Planilla_pruebas_modif.xlsx'
# # path1='Planilla_pruebas_modif2.xlsx'
# # path1='Planilla_pruebas_modif_sin_outliers.xlsx'

# Data_df = pd.read_excel(path1) 

# #convertir a numpy
# Data=Data_df.to_numpy()

# XX = Data[1:20,:] 

# XY = Data[21:28,:] 

# XZ = Data[29:,:]

# Data = np.concatenate((XX,XY,XZ),axis=0)

# X = Data[:,0:9]
# y = Data[:,9]

# # X=X[:,[0]]
# # X=X[:,[2]]
# # X=X[:,[4]]
# # X=X[:,[8]]
# X=X[:,[0,2]]
# # X=X[:,[2,4]]
# # X=X[:,[0,2,4]]
# # X=X[:,[2,4,8]]
# # X=X[:,[0,1,2,4]]
# # X=X[:,[0,2,4,8]]
# # X=X[:,[0,1,2,4,8]]

# # X=X[:,[1]]
# # X=X[:,[0,1]]
# # X=X[:,[0,4]]
# # X=X[:,[0,8]]
# # X=X[:,[1,2]]
# # X=X[:,[1,4]]
# # X=X[:,[1,8]]
# # X=X[:,[2,8]]
# # X=X[:,[4,8]]
# # X=X[:,[0,1,2]]
# # X=X[:,[0,1,4]]
# # X=X[:,[0,1,8]]
# # X=X[:,[0,2,8]]
# # X=X[:,[0,4,8]]
# # X=X[:,[1,2,4]]
# # X=X[:,[1,2,8]]
# # X=X[:,[1,4,8]]
# # X=X[:,[0,1,2,8]]
# # X=X[:,[0,1,4,8]]
# # X=X[:,[1,2,4,8]]



# # X=X[:,[2,4]]
# # #===========================================================

# # #============nombre variables modelos y resultados=====================
# models = []
# confusion_matrices_test=[]
# confusion_matrices_train=[]

# num_exp=100
# # #===========================================================

# #===========================================================
# # codificacion de la variable a clasificar

# #codificar salidas para diferenciar entre peligro de caida y no peligro de caida----

# y_codif=np.zeros([y.shape[0]])

# umbral=22 # 21 y 22 son los mejores

# for n in range(0,y.shape[0]):
    
#     if y[n] < umbral:
#         y_codif[n]=0       #hasta umbral -1
#     else:
#         y_codif[n]=1       #desde umbral 

# # #===========================================================

# for i in range(0,num_exp):

#     # #===========================================================
#     # Separar datos de entrenamiento y evaluacion
#     X_train, X_test, y_train, y_test = train_test_split(X, y_codif, test_size=0.3) #, random_state=27)
#     # #===========================================================
    
    
#     #-----------------------Clasificadores----------------------
    
#     #===========================================================
    
#     #-----------------------LogisticRegression----------------------
#     Clasif = LogisticRegression(C=0.1, class_weight = "balanced")
    
#     # #-----------------------RandomForest----------------------
#     # Clasif = RandomForestClassifier(max_depth=4, random_state=0)#, class_weight = "balanced")
    
    
#     #-----------------------SVM----------------------
#     # Crear el clasificador lineal o no lineal. 
#     # Si C disminuye permite incluir algunos errores de
#     # clasificacion a traves de un margen mayor, si C aumenta el margen disminuye para
#     # evitar errores de clasificacion
#     # gamma ,para RBF, define el radio de influencia de las muestras selecionadas como vectores de 
#     # soporte. Valores grandes de gamma determinan un radio de influencia chico,
#     # lo que implica seleccionar mas vectores de soporte y menos generalizacion. Valores
#     # pequeños implican menos vectores de soporte necesarios y mas generalizacion
    
    
#     # Clasif = svm.SVC(kernel="linear", C=10)
#     # Clasif = svm.SVC(kernel="linear", C=0.01, class_weight = "balanced")
#     # Clasif = svm.SVC(kernel="rbf")
#     # Clasif = svm.SVC(kernel="rbf", gamma=1)
#     # Clasif = svm.SVC(kernel="rbf", gamma=0.01, class_weight = "balanced", C=0.1)
#     # Clasif = svm.SVC(kernel="rbf", gamma=10, class_weight = "balanced")
#     # Clasif = svm.SVC(kernel="poly", C=0.01, coef0= 2, degree=2, gamma=0.1)
    
    
    
#     # #-----------------------Trees----------------------
#     # Clasif = tree.DecisionTreeClassifier(class_weight="balanced", min_samples_split=17)
#     # Clasif = tree.DecisionTreeClassifier(class_weight="balanced", ccp_alpha=0.1)
    
    
#     # # # # #-----------------------KNN----------------------
#     # n_neighbors = 10 #1,5,10,15
    
#     # weights="uniform"
#     # # weights="distance"
    
#     # Clasif = clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)


#     #entrenar y evaluar el clasif 
#     y_pred = Clasif.fit(X_train, y_train).predict(X_test)
    
#     y_pred_entrena = Clasif.predict(X_train)
    
#     #===========================================================
    
#     # ---------------evaluacion del entrenamiento-------------------
#     conf_mat_train=confusion_matrix(y_train, y_pred_entrena)
#     # ---------------evaluacion de generalizacion-------------------
#     conf_mat_test=confusion_matrix(y_test, y_pred)
    
#     #===========================================================

#     models.append(Clasif)
#     confusion_matrices_test.append(conf_mat_test)
#     confusion_matrices_train.append(conf_mat_train)
    
    
    # print(i)

# #============almacenar resultados=======================
# with open("models02_v3.pckl", "wb") as f:
#     for model in models:
#           pickle.dump(model, f)
# with open("matrices_conf_train02_v3.pckl", "wb") as f:
#     pickle.dump(confusion_matrices_train, f) 
# with open("matrices_conf_test02_v3.pckl", "wb") as f:
#     pickle.dump(confusion_matrices_test, f)                  
# #===========================================================         
#============leer resultados============================
models = []
tag_modelo = "KNN"
tag_base = "vuln_2020_2025_num+texto_num+texto"
#with open(f"pickle/models_{tag_modelo}_{tag_base}.pckl", "rb") as f:
#    while True:
#        try:
#            models.append(pickle.load(f))
#        except EOFError:
#            break

with open(f"pickle/matrices_conf_train_{tag_modelo}_{tag_base}.pckl", "rb") as f:
    confusion_matrices_train=pickle.load(f)
with open(f"pickle/matrices_conf_test_{tag_modelo}_{tag_base}.pckl", "rb") as f:
    confusion_matrices_test=pickle.load(f)         

# #============metricas acumuladas=======================
#===========================================================

n_experiments = 50 

for j in range(0,16):
    MA_train=np.zeros((2,2))
    MA_test=np.zeros((2,2))
    for i in range(j*n_experiments,(j*n_experiments)+n_experiments):
        MA_train=MA_train+confusion_matrices_train[i]
        MA_test=MA_test+confusion_matrices_test[i]

        mean_acuracy=(MA_test[0][0]+MA_test[1][1])/(np.sum(MA_test))
    mean_TPR=MA_test[0][0]/(MA_test[0][0]+MA_test[0][1])
    mean_TNR=MA_test[1][1]/(MA_test[1][0]+MA_test[1][1])
    mean_PREC_enf=MA_test[0][0]/(MA_test[0][0]+MA_test[1][0])
    mean_PREC_sano=MA_test[1][1]/(MA_test[0][1]+MA_test[1][1])
    mean_recall_enf=MA_test[0][0]/(MA_test[0][0]+MA_test[0][1])
    mean_recall_sano=MA_test[1][1]/(MA_test[1][0]+MA_test[1][1])
    mean_f1_enf = (2*(mean_PREC_enf * mean_recall_enf))/(mean_PREC_enf + mean_recall_enf)
    mean_f1_sano = (2*(mean_PREC_sano * mean_recall_sano)/(mean_PREC_sano + mean_recall_sano))

    class_mean_acuracy=0.5*(mean_recall_enf + mean_recall_sano)


    print('\n MC Train:'+str(j+1))
    print(MA_train)  
    print('\n MC Test:' + str(j+1))
    print(MA_test)   
    print('\n Accuracy: %.3f' % mean_acuracy)
    print('\n Accuracy x class: %.3f' % class_mean_acuracy)
    print('\n Recall no exploit: %.3f' % mean_TPR)
    print('\n Recall exploit: %.3f' % mean_TNR)
    print('\n Precision no exploit: %.3f' % mean_PREC_enf)
    print('\n Precision exploit: %.3f' % mean_PREC_sano)
    print('\n F1 Score no exploit: %.3f' % mean_f1_enf)
    print('\n F1 Score exploit: %.3f' % mean_f1_sano)
 

#===========================================================

# #Generar grafica de evaluacion de datos de generalizacion
# #se usan datos de entrenamiento para superficie
# disp = DecisionBoundaryDisplay.from_estimator(Clasif, X, response_method="predict", xlabel="X1", ylabel="X2", alpha=0.5)

# scatter = disp.ax_.scatter(X[:,0], X[:,1], c = y_codif, edgecolor="k")

# # produce a legend with the unique colors from the scatter
# legend1 = disp.ax_.legend(*scatter.legend_elements(), title="Classes")
# disp.ax_.add_artist(legend1)

# disp.ax_.legend()
# plt.xlabel("Education")
# plt.ylabel("TMT")
# plt.title("Clasificador opt en todos los datos")
# plt.show()


# #============metricas por iteracion=======================

# acuraccies=np.zeros(num_exp)

# for i in range(0,num_exp):
#     acuraccies[i]=(confusion_matrices_test[i][0,0]+confusion_matrices_test[i][1,1])/np.sum(confusion_matrices_test[i])
    
# mean_acuracy=np.mean(acuraccies)  
# std_acuracy=np.std(acuraccies)  

# print('\n Accuracy: %.3f' % mean_acuracy)
# print('\n std Accuracy: %.3f' % std_acuracy)
# print('\n MC Test:')
# print(MA_test)

# num_bins = 10
# fig, ax = plt.subplots()
# n, bins, patches = ax.hist(acuraccies, num_bins, density=False, alpha=.3, color ="blue", edgecolor='blue')
# ax.set_xlabel('Accuracy')
# ax.set_ylabel('Frequency')
# ax.set_title('Histogram: 'fr'$\mu={mean_acuracy:.2f}$, $\sigma={std_acuracy:.2f}$')
# fig.tight_layout()
# plt.show()
# #===========================================================

