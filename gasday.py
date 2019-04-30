import datetime
from collections import Sequence


class GasDay(datetime.date, Sequence):
    """Class for gas days

    gas days starts at 06:00 current day and ends 06:00 next day
    """

    def __new__(cls, time=datetime.datetime.now()):
        '''
        erzeugt ein neues Gastag Objekt

        :param time: Zeitpunkt für den der Gastag kreiert werden soll, wenn nicht angegeben
                 dann ist das des aktuelle Zeitpunkt
        :return: der Gastag zum jeweiligen Zeitpunkt (datetime.date)
        '''
        cls.offsetHours = 7
        time = time + datetime.timedelta(hours=-cls.offsetHours)
        return datetime.date.__new__(cls, time.year, time.month, time.day)


    def __init__(self, zeit=datetime.datetime.now()):
        zeit = zeit + datetime.timedelta(hours=-self.offsetHours)
        self.normalInitTime = zeit
        self.init_hour = zeit.hour

    def __str__(self):
        """
        Gastag als string.
        Returns
        -------
        string
            Gastag als String in der Form YYYYMMDD
        """
        return datetime.datetime.strftime(self, "%Y%m%d")

    def __getitem__(self, item):
        rval = []
        if isinstance(item, slice):
            start = item.start
            if None == start:
                start = 0
            stop = item.stop
            if None == stop:
                stop = 0
            len = stop - start + 1
            if len > 0:
                indices = item.indices(len)
                for i in range(*indices):
                    gdn = self[i]
                    rval.insert(i, gdn)
        else:
            rval = GasDay(datetime.datetime(self.year, self.month, self.day) + datetime.timedelta(days=item, hours=self.offsetHours))
        return rval

    def get_hour(self, item):
        """
        Zeitpunkt in Realzeit für Start und Ende der Gastagsstunde

        :param
        item die Gasstunden für die Start und Ende ermittelt werden sollen, Null indiziert

        :returns
        datetime.datetime, datetime.datetime
        von
            reale Zeit, zu der die Gasstunde beginnt
        bis
            reale Zeit, an der die Gasstunde endet
        """
        day = int(item / 24)
        h0 = datetime.datetime.combine(self, datetime.time(hour=self.offsetHours))
        for h in range(0, 24):
            hour_from = h0 + datetime.timedelta(days=day, hours=item - 1)
            hour_to = h0 + datetime.timedelta(days=day, hours=item)
        return hour_from, hour_to

    def get_all_hours(self):
        """
        Start und Ende aller Gastagsstunden in Realzeit

        Gibt die Start- und Endezeiten aller Gastagsstunden in Realzeit zurück

        :returns
        datetime.datetime[], datetime.datetime[]
        von
            reale Zeit, zu der die jeweilige Gasstunde beginnt
        bis
            reale Zeit, an der die jeweilige Gasstunde endet
        """
        hour_from = []
        hour_to = []
        for h in range(0, 24):
            f, t = self.get_hour(h)
            hour_from.insert(h, f)
            hour_to.insert(h, t)
        return hour_from, hour_to


    def get_init_time_hour(self):
        """
        Gasstunde, zu der das Object initialisiert wurde
        """
        return self.init_hour
