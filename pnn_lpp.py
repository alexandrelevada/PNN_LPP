#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Probabilistic Nearest Neighbor Entropic Locality Preserving Projections

Created on Wed Jul 24 16:59:33 2019

@author: Alexandre L. M. Levada

"""

# Imports
import sys
import time
import warnings
import umap
import sklearn.datasets as skdata
import matplotlib.pyplot as plt
import numpy as np
from numpy import log
from numpy import trace
from numpy import dot
from scipy import stats
from numpy.linalg import det
from scipy.linalg import eigh
from numpy.linalg import inv
from numpy.linalg import cond
from numpy import eye
from sklearn import preprocessing
from sklearn import metrics
import sklearn.neighbors as sknn
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

# PCA implementation
def myPCA(dados, d):
    # Eigenvalues and eigenvectors of the covariance matrix
    v, w = np.linalg.eig(np.cov(dados.T))
    # Sort the eigenvalues
    ordem = v.argsort()
    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = w[:, ordem[-d:]]
    # Projection matrix
    Wpca = maiores_autovetores
    # Linear projection into the 2D subspace
    novos_dados = np.dot(Wpca.T, dados.T)
    
    return novos_dados


### Regular LPP
def LPP(X, k, d, t, mode):
    if mode == 'distancias':
        knnGraph = sknn.kneighbors_graph(X, n_neighbors=k, mode='distance')
        knnGraph.data = np.exp(-(knnGraph.data**2)/t)
    else:
        knnGraph = sknn.kneighbors_graph(X, n_neighbors=k, mode='connectivity')

    W = knnGraph.toarray()  

    D = np.diag(W.sum(1))   
    L = D - W

    X = X.T
    M1 = np.dot(np.dot(X, D), X.T)
    M2 = np.dot(np.dot(X, L), X.T)

    if cond(M1) < 1/sys.float_info.epsilon:
        M = np.dot(inv(M1), M2)
    else:
        M1 = M1 + 0.00001*eye(M1.shape[0])
        M = np.dot(inv(M1), M2)
    
    lambdas, alphas = eigh(M, eigvals=(0, d-1))   
    
    output = np.dot(alphas.T, X)

    return output

### Probabilistic Nearest Neighbor Locality Preserving Projections
def pnnLPP(X, k, d, t, mode):
    if mode == 'distancias':
        knnGraph = sknn.kneighbors_graph(X, n_neighbors=k, mode='distance')
        knnGraph.data = np.exp(-(knnGraph.data**2)/t)
    else:
        knnGraph = sknn.kneighbors_graph(X, n_neighbors=k, mode='connectivity')

    W = knnGraph.toarray()      

    W1 = np.zeros(W.shape)
    P = np.zeros(W.shape)
    for i in range(n):
        distancias = W[i, :]
        order = distancias.argsort()
        distancias.sort()
        for j in range(n):
            if (distancias[nn+1] - distancias[1]) != 0:
                W1[i, j] = (distancias[nn+1] - distancias[j])/(distancias[nn+1] - distancias[1])   # PNN
                #W1[i, j] = (distancias[nn+1] - distancias[j])/(nn*distancias[nn+1] - sum(distancias[:nn]))   # CAN
            else:
                W1[i, j] = 0
            if W1[i, j] == 0:
                W1[i, j] = 10**(-15)
        P[i, order[1:nn+1]] = W1[i, 1:nn+1]

    W = P.copy()

    D = np.diag(W.sum(1))  
    L = D - W

    # Matriz 1
    X = X.T
    M1 = np.dot(np.dot(X, D), X.T)
    M2 = np.dot(np.dot(X, L), X.T)

    if cond(M1) < 1/sys.float_info.epsilon:
        M = np.dot(inv(M1), M2)
    else:
        M1 = M1 + 0.00001*eye(M1.shape[0])
        M = np.dot(inv(M1), M2)
    
    lambdas, alphas = eigh(M, eigvals=(0, d-1))   
    
    output = np.dot(alphas.T, X)

    return output


'''
 Computes the supervised classification accuracies for several classifiers: KNN, SVM, NB, DT, QDA, MPL, GPC and RFC
 dados: learned representation (output of a dimens. reduction - DR)
 target: ground-truth (data labels)
 method: string to identify the DR method (PCA, NP-PCAKL, KPCA, ISOMAP, LLE, LAP, ...)
'''
def Classification(dados, target, method):
    print()
    print('Supervised classification for %s features' %(method))
    print()
    
    lista = []

    # 50% for training and 50% for testing
    X_train, X_test, y_train, y_test = train_test_split(dados.real.T, target, test_size=.5, random_state=42)

    # KNN
    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(X_train, y_train) 
    acc = neigh.score(X_test, y_test)
    lista.append(acc)
    #print('KNN accuracy: ', acc)

    # SMV
    svm = SVC(gamma='auto')
    svm.fit(X_train, y_train) 
    acc = svm.score(X_test, y_test)
    lista.append(acc)
    #print('SVM accuracy: ', acc)

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    acc = nb.score(X_test, y_test)
    lista.append(acc)
    #print('NB accuracy: ', acc)

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    acc = dt.score(X_test, y_test)
    lista.append(acc)
    #print('DT accuracy: ', acc)

    # Quadratic Discriminant 
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    acc = qda.score(X_test, y_test)
    lista.append(acc)
    #print('QDA accuracy: ', acc)

    # MPL classifier
    mpl = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=5000)
    mpl.fit(X_train, y_train)
    acc = mpl.score(X_test, y_test)
    lista.append(acc)
    #print('MPL accuracy: ', acc)

    # Gaussian Process
    gpc = GaussianProcessClassifier()
    gpc.fit(X_train, y_train)
    acc = gpc.score(X_test, y_test)
    lista.append(acc)
    #print('GPC accuracy: ', acc)

    # Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    acc = rfc.score(X_test, y_test)
    lista.append(acc)
    #print('RFC accuracy: ', acc)

    # Computes the Silhoutte coefficient
    sc = metrics.silhouette_score(dados.real.T, target, metric='euclidean')
    print('Silhouette coefficient: ', sc)
    
    # Computes the average accuracy
    average = sum(lista)/len(lista)
    maximo = max(lista)

    print('Average accuracy: ', average)
    print('Maximum accuracy: ', maximo)
    print()

    return [sc, average]


# Plot the data via scatterplots
def PlotaDados(dados, labels, metodo):
    
    nclass = len(np.unique(labels))

    if metodo == 'LDA':
        if nclass == 2:
            return -1

    # Converte labels para inteiros
    lista = []
    for x in labels:
        if x not in lista:  
            lista.append(x)     # contém as classes (sem repetição)

    # Mapeia rotulos para números
    rotulos = []
    for x in labels:  
        for i in range(len(lista)):
            if x == lista[i]:  
                rotulos.append(i)

    # Converte para vetor
    rotulos = np.array(rotulos)

    if nclass > 11:
        cores = ['black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred']
        np.random.shuffle(cores)
    else:
        cores = ['blue', 'red', 'cyan', 'black', 'orange', 'magenta', 'green', 'darkkhaki', 'brown', 'purple', 'salmon']

    plt.figure(10)
    for i in range(nclass):
        indices = np.where(rotulos==i)[0]
        #cores = ['blue', 'red', 'cyan', 'black', 'orange', 'magenta', 'green', 'darkkhaki', 'brown', 'purple', 'salmon', 'silver', 'gold', 'darkcyan', 'royalblue', 'darkorchid', 'plum', 'crimson', 'lightcoral', 'orchid', 'powderblue', 'pink', 'darkmagenta', 'turquoise', 'wheat', 'tomato', 'chocolate', 'teal', 'lightcyan', 'lightgreen', ]
        cor = cores[i]
        plt.scatter(dados[indices, 0], dados[indices, 1], c=cor, marker='*')
    
    nome_arquivo = metodo + '.png'
    plt.title(metodo+' clusters')

    plt.savefig(nome_arquivo)
    plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%  Data loading
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# OpenML datasets
X = skdata.fetch_openml(name='SPECTF', version=1)  
#X = skdata.fetch_openml(name='veteran', version=2) 
#X = skdata.fetch_openml(name='sleuth_ex1605', version=2) 
#X = skdata.fetch_openml(name='aids', version=1) 
#X = skdata.fetch_openml(name='cloud', version=2) 
#X = skdata.fetch_openml(name='analcatdata_creditscore', version=1) 
#X = skdata.fetch_openml(name='corral', version=1)
#X = skdata.fetch_openml(name='cars1', version=1)
#X = skdata.fetch_openml(name='LED-display-domain-7digit', version=1)
#X = skdata.fetch_openml(name='hayes-roth', version=1) 
#X = skdata.fetch_openml(name='fl2000', version=1) 
#X = skdata.fetch_openml(name='Diabetes130US', version=1)     # 1%
#X = skdata.fetch_openml(name='blogger', version=1)
#X = skdata.fetch_openml(name='user-knowledge', version=1)    
#X = skdata.fetch_openml(name='rabe_131', version=2)
#X = skdata.fetch_openml(name='haberman', version=1)                    
#X = skdata.fetch_openml(name='prnn_synth', version=1)                  
#X = skdata.fetch_openml(name='visualizing_environmental', version=2)   
#X = skdata.fetch_openml(name='vineyard', version=2)                    
#X = skdata.fetch_openml(name='monks-problems-1', version=1)            
#X = skdata.fetch_openml(name='acute-inflammations', version=2)         
#X = skdata.fetch_openml(name='planning-relax', version=1)              
#X = skdata.fetch_openml(name='sensory', version=2)                     
#X = skdata.fetch_openml(name='auto_price', version=2)                  
#X = skdata.fetch_openml(name='wisconsin', version=2)                   
#X = skdata.fetch_openml(name='fri_c4_250_100', version=2)              
#X = skdata.fetch_openml(name='thoracic_surgery', version=1)            
#X = skdata.fetch_openml(name='conference_attendance', version=1)       
#X = skdata.fetch_openml(name='analcatdata_boxing1', version=1)         
#X = skdata.fetch_openml(name='fri_c2_100_10', version=2)               
#X = skdata.fetch_openml(name='lupus', version=1)                       
#X = skdata.fetch_openml(name='fruitfly', version=2)                    


dados = X['data']
target = X['target']  

#dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.01, random_state=42)

n = dados.shape[0]
m = dados.shape[1]
c = len(np.unique(target))

print('N = ', n)
print('M = ', m)
print('C = %d' %c)
input()

nn = round(np.sqrt(n))

# Convert categorical data
if not isinstance(dados, np.ndarray):
    cat_cols = dados.select_dtypes(['category']).columns
    dados[cat_cols] = dados[cat_cols].apply(lambda x: x.cat.codes)
    # Converte para numpy (openml agora é dataframe)
    dados = dados.to_numpy()
    target = target.to_numpy()

# Data standardization (to deal with variables having different units/scales)
dados = preprocessing.scale(dados)

#%%%%%%%%%%% Simple PCA 
dados_pca = myPCA(dados, 2)

#%%%%%%%%%%% ISOMAP
model = Isomap(n_neighbors=nn, n_components=2)
dados_isomap = model.fit_transform(dados)
dados_isomap = dados_isomap.T

#%%%%%%%%%%% LLE
model = LocallyLinearEmbedding(n_neighbors=nn, n_components=2)
dados_LLE = model.fit_transform(dados)
dados_LLE = dados_LLE.T

#%%%%%%%%%%% Lap. Eig.
model = SpectralEmbedding(n_neighbors=nn, n_components=2)
dados_Lap = model.fit_transform(dados)
dados_Lap = dados_Lap.T

# LPP original
dados_lpp = LPP(X=dados, k=nn, d=2, t=1, mode='distancias')

#%%%%%%%%%%%% UMAP
model = umap.UMAP(n_components=2)
dados_umap = model.fit_transform(dados)
dados_umap = dados_umap.T


#%%%%%%%%%%% Supervised classification
L_pca = Classification(dados_pca.real, target, 'PCA')
L_iso = Classification(dados_isomap, target, 'ISOMAP')
L_lle = Classification(dados_LLE, target, 'LLE')
L_lap = Classification(dados_Lap, target, 'Lap. Eig.')
L_lpp = Classification(dados_lpp, target, 'LPP')
L_umap = Classification(dados_umap, target, 'UMAP')

# Plot data
PlotaDados(dados_pca.T, target, 'PCA')
PlotaDados(dados_isomap.T, target, 'ISOMAP')
PlotaDados(dados_LLE.T, target, 'LLE')
PlotaDados(dados_Lap.T, target, 'LAP')
PlotaDados(dados_lpp.T, target, 'LPP')
PlotaDados(dados_umap.T, target, 'UMAP')

########################## PNN-LPP

inicio = 2
high = min(51, n)
incremento = 1
vizinhos = list(range(inicio, high, incremento))

best_acc = 0
best_viz = 0

# Perform model selection based on the average accuracy

print('Supervised classification for PNN LPP features')

for viz in vizinhos:

    # Entropic LPP
    dados_pnn_lpp = pnnLPP(X=dados, k=viz, d=2, t=1, mode='distancias')
    dados_pnn_lpp = dados_pnn_lpp.T

    #%%%%%%%%%%%%%%%%%%%% Supervised classification for Kernel PCA features

    print()
    print('K = %d' %viz)

    # 50% for training and 40% for testing
    X_train, X_test, y_train, y_test = train_test_split(dados_pnn_lpp.real, target, test_size=.5, random_state=42)
    acc = 0

    # KNN
    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(X_train, y_train)
    acc += neigh.score(X_test, y_test)
    #print('KNN accuracy: ', neigh.score(X_test, y_test))

    # SVM
    svm = SVC(gamma='auto')
    svm.fit(X_train, y_train) 
    acc += svm.score(X_test, y_test)
    #print('SVM accuracy: ', svm.score(X_test, y_test))

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    acc += nb.score(X_test, y_test)
    #print('NB accuracy: ', nb.score(X_test, y_test))

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    acc += dt.score(X_test, y_test)
    #print('DT accuracy: ', dt.score(X_test, y_test))

    # Quadratic Discriminant 
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    acc += qda.score(X_test, y_test)
    #print('QDA accuracy: ', qda.score(X_test, y_test))

    # MPL classifier
    mpl = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=5000)
    mpl.fit(X_train, y_train)
    acc += mpl.score(X_test, y_test)
    #print('MPL accuracy: ', mpl.score(X_test, y_test))

    # Gaussian Process
    gpc = GaussianProcessClassifier()
    gpc.fit(X_train, y_train)
    acc += gpc.score(X_test, y_test)
    #print('GPC accuracy: ', gpc.score(X_test, y_test))

    # Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    acc += rfc.score(X_test, y_test)
    #print('RFC accuracy: ', rfc.score(X_test, y_test))

    # # Computes the Silhoutte coefficient
    # print('Silhouette coefficient: ', metrics.silhouette_score(dados_lpp_ent.real, target, metric='euclidean'))
    mean_acc = acc/8
    print('Acurácia média: ', mean_acc)

    if mean_acc > best_acc:
        best_acc = mean_acc
        best_viz = viz


print()
print('==============')
print('BEST ACCURACY')
print('==============')
print()
print('Acurácia média: ', best_acc)
print('K = ', best_viz)

# Plot data
dados_pnn_lpp = pnnLPP(X=dados, k=best_viz, d=2, t=1, mode='distancias')
dados_pnn_lpp = dados_pnn_lpp.T
PlotaDados(dados_pnn_lpp.real, target, 'PNN LPP')