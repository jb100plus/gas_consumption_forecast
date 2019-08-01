import pandas as pd
import gasprognoseConstants
import logger
from prognosebase import PrognoseBase
import sys
import seaborn as sns
import matplotlib.pyplot as plt



if __name__ == '__main__':
    log = logger.Logger()

    dotraining = True

    if len(sys.argv) == 2:
        arg = sys.argv[1]
        if arg == 'validate':
            dotraining = False

    # during training data will be divided: 80% for training 20% for testing
    train_data = pd.read_csv('DAT_TEMP_RLM_SLP_KORR.csv', sep=";", decimal=",", thousands='.',
                             parse_dates=['DAT'], dayfirst=True, index_col=['DAT'])[-(+365+365):]

    train_data['WT'] = train_data.index.to_series().apply(gasprognoseConstants.weekday)
    train_data['MONAT'] = train_data.index.to_series().apply(gasprognoseConstants.month)
    train_data['SEASON'] = train_data['MONAT'].apply(gasprognoseConstants.season)
    train_data['QUARTAL'] = train_data['MONAT'].apply(gasprognoseConstants.quartal)
    train_data['WINTER'] = train_data['MONAT'].apply(gasprognoseConstants.winter)




    #train_data = pd.read_csv('DAT_TEMP_EV_STUNDE.csv', sep=";", decimal=",", thousands='.', parse_dates=['DAT'])[-365*24:]
    #train_data = train_data.groupby('DATH4').mean()
    #train_data['RLMEEFW'] = train_data['RLMEEFW'] * 4
    # define the columns that you want to use as features for training AND prediction
    # a json file with the limits is written
    features = dict()
    features['TEMP'] = {'column': 'TEMP'}
    features['TEMPGL'] = {'column': 'TEMPGL'}
    features['WT'] = {'column': 'WT'}
    features['WINTER'] = {'column': 'WINTER'} # ??
    features['SEASON'] = {'column': 'SEASON'}
    features['MONAT'] = {'column': 'MONAT'}
    features['QUARTAL'] = {'column': 'QUARTAL'}
    # define the model and the label
    rlmmodel = PrognoseBase(gasprognoseConstants.RLMMODEL, 'RLMKORREE')
    if dotraining:
        testdata = rlmmodel.trainmodel(train_data, features, showhistory=False)
        #stat = rlmmodel.compare(testdata, showplot=False)
    data = train_data[-365:]
    stat = rlmmodel.compare(data, showplot=True)
    stat['T'] = data['TEMP'].values
    sns.pairplot(stat[['T', 'Y', 'D', 'A']], diag_kind="kde")
    plt.show()

    exit(0)

    #rlmmodel.model.summary()
    slpmodel = PrognoseBase(gasprognoseConstants.SLPMODEL, 'ALLOKSLP')
    if dotraining:
        testdata = slpmodel.trainmodel(train_data, features, showhistory=False)
        stat = slpmodel.compare(testdata, showplot=False)
    #stat = slpmodel.compare(train_data[-365:], showplot=False)

    #stat['GASTAG'] = [datetime.date.fromtimestamp(i / 1000000000.0) for i in stat['I']]
    #print(stat[['GASTAG', 'Y', 'P', 'D', 'A']])
    exit(0)

