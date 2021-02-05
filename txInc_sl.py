# -*- coding: utf-8 -*-
"""
Affichage graphique du taux d'incidence - Webapp Streamlit

@author: Cédric LEBOCQ
"""

# %% Imports

from datetime import datetime
from numpy import trapz
import os.path
import glob
import pandas as pd  # conda install xlrd is necessary...
import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st


# %% Streamlit stuff

st.title("Taux d'incidence et positivité communes du dept. 13")


st.markdown("""
Auteur : **Cédric Lebocq**            
Cette application récupére les données directement sur le site du **gouvernement**
* **Data source:** [données laboratoires](https://www.data.gouv.fr/fr/datasets/donnees-de-laboratoires-infra-departementales-durant-lepidemie-covid-19/).
* Pour l'instant seules les communes de **0 à 10 000 habitants** peuvent être séléctionnées
* La population repose sur les données du recensement **2014**
""")

st.sidebar.header('Choix utilisateur')

# %% Charge les communes et la population


@st.cache
def load_static_data():
    #dsC = pd.read_csv("./data/communes2020.csv")
    dsP = pd.read_excel("./data/obs.terr_populationcom.xls", "Data")
    #return dsC, dsP
    return dsP


dsPop = load_static_data()

# %% Récupére les données actualisées doncernant le tx incidence sur data.gouv
# et réalise une mise en forme
# il s'agit du fichier sg-com-opendata

now = datetime.now()
dt_string = now.strftime("%Y-%m-%d")

# Cleanup
for CleanUp in glob.glob('./data/*.csv'):
    if not CleanUp.endswith(f'{dt_string}.csv'):    
        os.remove(CleanUp)
#


# si le fichier est déjà présent on ne le récupére pas à nouveau
if not os.path.isfile(f'./data/{dt_string}.csv'):
    # récupération du fichier brut
    url = 'https://www.data.gouv.fr/fr/datasets/r/c2e2e844-9671-4f81-8c81-1b79f7687de3'
    r = requests.get(url, allow_redirects=True)
    with open('./data/down.csv', 'wb') as dcom:
        dcom.write(r.content)

    # formatage
    with open('./data/down.csv', 'r') as inp, open(f'./data/{dt_string}.csv', 'w') as outp:
        header = True
        for line in inp:
            if header:
                newline = line.replace(';', ',')
                header = False
                outp.write(newline)
            else:
                chunks = line.split(';')
                if len(chunks) == 9:
                    newline = ','.join(chunks[0:3]) + ',' + \
                        ';'.join(chunks[3:5]) + ',' + \
                        ';'.join(chunks[5:7]) + ',' + \
                        ';'.join(chunks[7:])
                    outp.write(newline)

    os.remove('./data/down.csv')
# %% On charge le fichier dans un dataset


@st.cache
def load_cached_txInc():
    dsTi = pd.read_csv(f'./data/{dt_string}.csv', dtype={'com2020': object})
    return dsTi


dsTxInc = load_cached_txInc()


# %% Streamlit stuff

# Sidebar - fourchette de population
popMin, popMax = st.sidebar.slider('Fourchette Population', 0, 10000, value=[
                                   5000, 7000], key='pop', step=100)
st.markdown(
    f"Votre séléction de villages porte sur la fourchette de **{popMin}** à **{popMax}** habitants")

# %% On filtre les communes dans la fourchette (population de 2014)
com = dsPop[dsPop['Code'].astype(str).str.startswith('13') & (
    dsPop['p14_pop'] >= popMin) & (dsPop['p14_pop'] < popMax)].sort_values(by=['com2016'])
comName = com['com2016']           # contient le nom des communes


# %% Streamlit stuff
# Sidebar - Selection des villes dans la fourchette

selected_towns = st.sidebar.multiselect(
    'Villes', options=list(comName), default=list(comName))

com = com[com['com2016'].isin(selected_towns)][['Code', 'com2016', 'p14_pop']]
com.columns = ['Code', 'Nom', 'Population']
com.reset_index(inplace=True, drop=True)


st.dataframe(com[['Nom', 'Population']])


# %% Préparation du dataset des incidences

# on Filtre le dataset
dsTxInc13eq = dsTxInc[dsTxInc['com2020'].isin(com['Code'].astype(str))]


# Classes de taux de positivité
dicClsTp = {'0': 0.,
          '[1;5[': 5,
          '[5;10[': 10,
          '[10;15[': 15,
          '[15;20[': 20}

# et couleurs associées
dicCol = {0 :  'green',
          5:   'lightsalmon',
          10 : 'salmon',
          15 : 'firebrick',
          20 : 'saddlebrown'}

# Classes d'incidence
dicCls = {'[0;10[': 10,
          '[10;20[': 20,
          '[20;50[': 50,
          '[50;150[': 150,
          '[150;250[': 250,
          '[250;500[': 500,
          '[500;1000[': 1000,
          '[1000;Max]': 1500}


dsTxInc13eq = dsTxInc13eq.copy()
# mapping pour remplacer les classes par une valeur
dsTxInc13eq['ti'] = dsTxInc13eq['ti_classe'].map(dicCls)
dsTxInc13eq['tp'] = dsTxInc13eq['tp_classe'].map(dicClsTp)

# on ne conserve qu'une partie de la date dans la semaine glissante (la date de fin)
dsTxInc13eq['date'] = dsTxInc13eq['semaine_glissante'].str.slice(
    11, 21).astype('datetime64')
# mapping de substituion dans le dataset du tx d'incidence
dsCodeCom = com[['Code', 'Nom']]
dsCodeCom['Code'] = dsCodeCom['Code'].astype(str)
dicCodeCom = dict(zip(dsCodeCom['Code'], dsCodeCom['Nom']))
dsTxInc13eq['commune'] = dsTxInc13eq['com2020'].map(dicCodeCom)
dsTxInc13eqFiltered = dsTxInc13eq[['date', 'commune', 'clage_65', 'ti','tp']]

# Classe des -65 ans
dsInf65 = dsTxInc13eqFiltered[dsTxInc13eqFiltered['clage_65'] == 0][[
    'date', 'commune', 'ti','tp']]
# Classe des +65 ans
dsSup65 = dsTxInc13eqFiltered[dsTxInc13eqFiltered['clage_65'] == 65][[
    'date', 'commune', 'ti','tp']]

# pivot des dataframe pour le taux d'incidence
dsSup65pv = dsSup65.pivot_table(
    index='date', columns='commune', values='ti')  # +65 ans
dsInf65pv = dsInf65.pivot_table(
    index='date', columns='commune', values='ti')  # -65 ans

# pivot des dataframe pour le taux de positivité
dsSup65tp = dsSup65.pivot_table(
    index='date', columns='commune', values='tp')  # +65 ans
dsInf65tp = dsInf65.pivot_table(
    index='date', columns='commune', values='tp')  # -65 ans

# %% Fonction pour tracer graphique

def plotTxInc(dsInc, dsTp, resample, method, ylabell,ylabelr, legend):
    keeped = dsInc.columns

    # calcul du nombre de lignes et colonnes en fonction du nombre de village
    if len(keeped) % 2 == 0:
        ncols = 2
        nrows = len(keeped) // ncols
    else:
        ncols = 3
        if len(keeped) % 3 == 0:
            nrows = (len(keeped) // ncols)
        else:
            nrows = (len(keeped) // ncols) + 1

    # définition des subplots
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, nrows*3), sharey='all')

    row = 0
    col = 0

    ax2l=[]
    for com_ in keeped:
        tsres = dsInc[com_].resample(resample)
        tsint = tsres.interpolate(method=method)
        txMean = int(tsint.mean())

        if com_ == 'SAINT-MITRE-LES-REMPARTS':
            c = 'blue'
            lw = 3
        else:
            c = 'darkgreen'
            lw = 2
        
        if nrows > 1:
            ax1 = axs[row, col]
        else:
            ax1 = axs[col]
            
        ax2 = ax1.twinx()
        # afin de pouvoir partager la même echelle sur tous axe y secondaires
        ax2l.append(ax2)

        # Taux de positivité
        colors = [dicCol[int(key)] for key in dsTp[com_]]    
        ax2.bar(dsTp.index,dsTp[com_],alpha=0.3,color=colors)
        ax2.set(xlabel='date', ylabel=ylabelr)

        # Taux incidence
        ax1.plot(tsint.index, tsint, label=com_,
                           alpha=1, color=c, linewidth=lw)
        
        #moyenne
        ax1.axhline(
            txMean, 0, 1, label=f"Moyenne = {txMean}", alpha=1, color="orange", linewidth=lw)
        ax1.legend(loc='upper left', frameon=False)


        col += 1
        if col > (ncols-1):
            row += 1
            col = 0

    #partage de la meme echelle sur les axes y secondaires
    ax2ref = ax2l[0]
    for ax2share in ax2l:
        ax2ref.get_shared_y_axes().join(ax2ref,ax2share)

    for ax in axs.flat:
        ax.set(xlabel='date', ylabel=ylabell)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    fig.autofmt_xdate()
    fig.suptitle(legend, fontsize=14, y=0.95 - (0.005 * nrows), c='red')
    st.pyplot(fig)


# %% Fonction pour tracer la surface d'incidence
def plotSInc(sInc, legend):
    fig, ax = plt.subplots(figsize=(8, 8))
    sInc.plot.barh(x='Ville', y='Surface Incidence', ax=ax)
    if 'SAINT-MITRE-LES-REMPARTS' in sInc['Ville'].unique():
        idx = int(sInc[sInc['Ville'] == 'SAINT-MITRE-LES-REMPARTS'].index[0])
        ax.get_children()[idx].set_color('r')
    ax.legend([legend], loc='lower right')
    st.pyplot(fig)

# %% Fonction pour calculer la surface d'incidence


@st.cache
def getSInc(ds, resample, method):
    keeped = ds.columns
    areas = dict()
    for com_ in keeped:
        tsres = ds[com_].resample(resample)
        tsint = tsres.interpolate(method=method)

        area = trapz(tsint, dx=1)
        areas[com_] = np.ceil(area/4)

    txAreas = pd.DataFrame(areas.items(), columns=[
                           'Ville', 'Surface Incidence'])
    txAreas = txAreas.sort_values(by=['Surface Incidence'])
    txAreas = txAreas.reset_index(drop=True)
    MaxValue = txAreas['Surface Incidence'].max()
    
    #Normalisation
    txAreas['Surface Incidence'] = (txAreas['Surface Incidence'] / MaxValue)*100

    return txAreas

# %% Affichage des graphes

dsInf65tp = dsInf65tp.fillna(0)
dsSup65tp = dsSup65tp.fillna(0)

plotTxInc(dsInf65pv,dsInf65tp, '6H', 'cubic', 'tx inc.','tx pos. %', "Taux incidence -65 ans")
plotTxInc(dsSup65pv,dsSup65tp, '6H', 'cubic', 'tx inc.','tx pos. %', "Taux incidence +65 ans")


sIncInf65 = getSInc(dsInf65pv, '6H', 'linear')
sIncSup65 = getSInc(dsSup65pv, '6H', 'linear')

st.markdown("""
Les graphes ci-dessous représentent avec quelle **'force'** ont été frappés les village sur le mois
courant en prenant pour référence le village le plus touché (base 100).
Ce calcul est réalisé par intégration de la surface sous les courbes d'incidence. 
""")

plotSInc(sIncInf65, "Impact sur un mois -65ans")
plotSInc(sIncSup65, "Impact sur un mois +65ans")
