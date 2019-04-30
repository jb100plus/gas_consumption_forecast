import datetime
import json
import pandas as pd
import gasday
import gasprognoseConstants
import logger
from prognosebase import PrognoseBase


def write_csv(preds, filename, desc):
    tagstr = datetime.datetime.strftime(datetime.date.today(), '%Y%m%d')
    count = len(preds)

    gd = gasday.GasDay()

    gds = gd[1: count + 1]
    file = open(filename, 'w', encoding='utf-8')
    file.write('Datum der Auswertung:;' + tagstr + '\n')
    file.write('Zählpunkte:;' + desc + '\n')
    file.write('Einheit:;kWh' + '\n')
    i = 0
    for d in gds:
        w = int(round(preds[i][0] / 24, 0))
        v, b = d.get_all_hours()
        for h in b:
            file.write(datetime.datetime.strftime(h, '%d.%m.%Y %H:%M') + ';' + str(w) + '\n')
        i += 1
    file.close()
    log.logger.info(f'written {desc} for {count} days to {filename}')
    return 0

# fill json for history
def write_prognose_json(preds, filename, desc):
    prognosen = dict()
    try:
        with open(filename, 'r') as pj:
            prognosen = json.load(pj)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        log.logger.warning('Keine prognosen-datei gefunden ' + str(e))
    gd0 = gasday.GasDay()
    i = 0
    for gd in gd0[1: 6]:
        prognose_day = dict()
        try:
            prognose_day = prognosen[str(gd)]
        except KeyError as e:
            pass
        prognose_day[desc] = int(round(preds[i][0] / 24, 0))
        prognosen[str(gd)] = prognose_day
        i += 1
    with open(filename, 'w') as pj:
        json.dump(prognosen, pj)
    return 0


def log_summary(dats, pr, ps, rlmkorrkwh):
    for d, r, s, k in zip(dats, pr, ps, rlmkorrkwh):
        log.logger.info(str(d) + ' RLM kW: ' + str(round(r[0] / 24, 0)) + ' SLP kW: ' + str(round(s[0] / 24, 0))
                        + ' Summe kW:' + str(round((s[0] + r[0]) / 24, 0))
                        + '   RLM Korrektur kW:' + str(round(k / 24, 0)))


def getslpallokationd1(gdstr):
    alloktagnetz = dict()
    allokslp = None
    try:
        with open(gasprognoseConstants.ALLOKNETZJSON, 'r') as wp:
            alloktagnetz = json.load(wp)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        log.logger.warning('Keine SLP Allokationdatei gefunden ' + str(e))
    try:
        allokslp = alloktagnetz[gdstr]
        allokslp = allokslp['SLP']
        if allokslp > 0.0:
            log.logger.info('Benutze SLP Allokation für ' + gdstr)
        else:
            log.logger.info('unplausible Daten für SLP Allokation für ' + gdstr + ' :' + str(allokslp))
            allokslp = None
    except KeyError as e:
        log.logger.warning('Keine SLP Allokation gefunden ' + str(e))
    return allokslp


if __name__ == '__main__':
    log = logger.Logger()
    prognosedaten = pd.read_csv(gasprognoseConstants.PROGNOSEDATENCSV, sep=';', decimal=',')
    predictiondays = prognosedaten['DAT']
    rlmkorrkwdays = prognosedaten['RLMKORRKW']
    gd = gasday.GasDay()
    log.logger.info(str(gd) + ' ' + 64 * '-')
    if str(predictiondays[0]) != str(gd[1]):
        log.logger.error(
            'Keine Prognosedaten für Gastag ' + str(gd[1]) + ' gefunden; Daten ab: ' + str(predictiondays[0]))
        exit(1)
    # y: label
    slpmodel = PrognoseBase(gasprognoseConstants.SLPMODEL, 'ALLOKSLP')
    rlmmodel = PrognoseBase(gasprognoseConstants.RLMMODEL, 'RLMKORREE')
    # get normalized data
    prog_x = slpmodel.get_prediction_data(prognosedaten)
    ps = slpmodel.predict(prog_x)
    pr = rlmmodel.predict(prog_x)
    # adjust with correction data in kW -> calculate for a whole day * 24
    # TODO: adjust for summer winter time
    korrkwh = [v * 24 for v in rlmkorrkwdays]
    pr = [sum(pp) for pp in zip(pr, korrkwh)]
    # usd SLP values if available
    slpkWh = getslpallokationd1(str(gd[1]))
    if slpkWh is not None:
        tfslp = f"tf slp incl OOA: {round(ps[0][0]):7.0f} kWh -> {round(ps[0][0] / 24, 0):6.0f} kW"
        anslp = f"an slp excl OOA: {round(slpkWh):7.0f} kWh -> {round(slpkWh / 24, 0):6.0f} kW"
        slpkWhhr = round(slpkWh * gasprognoseConstants.ALLOKANOOARATIO, 0)
        anslphr = f"an slp incl OOA: {slpkWhhr:7.0f} kWh -> {round(slpkWhhr / 24, 0):6.0f} kW"
        ps[0][0] = slpkWhhr  # override tensorflow forecast
        log.logger.info(tfslp)
        log.logger.info(anslp)
        log.logger.info(anslphr)

    write_csv(ps, gasprognoseConstants.SLPPROGNOSEFILE, 'SLP')
    write_csv(pr, gasprognoseConstants.RLMPROGNOSEFILE, 'RLM')
    rlmkorrkwh = (k * 24 for k in rlmkorrkwdays)
    log_summary(predictiondays, pr, ps, rlmkorrkwh)
    # save forecasts for history
    write_prognose_json(ps, gasprognoseConstants.PROGNOSENJSON, 'SLP')
    write_prognose_json(pr, gasprognoseConstants.PROGNOSENJSON, 'RLM')
    exit(0)
