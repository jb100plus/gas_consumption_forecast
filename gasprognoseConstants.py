import os
import datetime
#OUTPUTPATH = '/var/samba/gas/'
#LOGPATH = '/var/samba/gas/'
OUTPUTPATH = './data/'
LOGPATH = './data/'
# also in productive system
DATAPATH = './data/'

WETTERPROGNOSEJSON = os.path.join(DATAPATH, 'wetterprognosen.json')
PROGNOSEDATENJSON = os.path.join(DATAPATH, 'prognosedaten.json')
ALLOKNETZJSON = os.path.join(DATAPATH, 'alloktagnetz.json')
PROGNOSENJSON = os.path.join(DATAPATH, 'prognosen.json')
GASPOOLJSON = os.path.join(DATAPATH, 'gaspool.json')

MAILBOXDATAXML = os.path.join(OUTPUTPATH, 'mailboxdata.xml')
PROGNOSEDATENCSV = os.path.join(OUTPUTPATH, 'PROG_DATA.CSV')
RLMPROGNOSEFILE = os.path.join(OUTPUTPATH, 'Aschersleben_Prognose_RLM.CSV')
SLPPROGNOSEFILE = os.path.join(OUTPUTPATH, 'Aschersleben_Prognose_SLP.CSV')

SLPMODEL = 'gasprognose_SLP_model'
RLMMODEL = 'gasprognose_RLM_model'

LOGFILE = os.path.join(LOGPATH, 'gasprognose.log')

#accounts moved to secrets.py

ALLOKANOOARATIO = 1.40
RLMKORRKW = 4000.0 #2 Module elektrisch

# x 1..12
is_valid_month = lambda x: 1 <= x <= 12
quartal = lambda x: None if not is_valid_month(x) else int((x + 2) / 3)
season = lambda x: None if not is_valid_month(x) else 2 if x in (1, 2, 3, 11, 12) else 1 if x in (4, 5, 9, 10) else 0
winter = lambda x: None if not is_valid_month(x) else 1 if not x in (4, 5, 6, 7, 8, 9) else 0
month = lambda x: x.month
dateFromInt = lambda x:  None if x < 20000101 or x > 99990101 else datetime.date(int(str(x)[0:4]), int(str(x)[4:6]), int(str(x)[6:8]))
dateToInt = lambda x: x.year * 10000 + x.month * 100 + x.day
# excel like Sunday: 1 ... Saturday: 7
weekday = lambda x: (x.weekday() + 1) % 7 + 1
