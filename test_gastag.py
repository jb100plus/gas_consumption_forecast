import unittest
import gasday
import datetime
import os
import gasprognoseConstants
#import trainmodel

class TestGastag(unittest.TestCase):
    '''
    def test_gastag(self):
        t = datetime.datetime.now()
        tstr = datetime.datetime.strftime(t + datetime.timedelta(hours=-7), "%Y%m%d")
        gd = gasday.GasDay(t)
        self.assertEqual(str(gd), tstr)
        self.assertEqual(str(gd[0]), tstr)
        self.assertGreater(str(gd[1]), tstr)
        self.assertLess(str(gd[-1]), tstr)
        #self.fail()

    def test_doedel(self):
        self.assertEqual(1,1)


    def test_train(self):
        ec = os.system("python trainmodel.py validate")
        self.assertEqual(ec, 0)
        print(ec)
    '''

    def test_lambdas(self):
        self.assertEqual(gasprognoseConstants.is_valid_month(13), False)
        self.assertEqual(gasprognoseConstants.is_valid_month(0), False)
        self.assertEqual(gasprognoseConstants.is_valid_month(1), True)
        self.assertEqual(gasprognoseConstants.is_valid_month(12), True)
        self.assertEqual(gasprognoseConstants.quartal(1), 1)
        self.assertEqual(gasprognoseConstants.quartal(2), 1)
        self.assertEqual(gasprognoseConstants.quartal(3), 1)
        self.assertEqual(gasprognoseConstants.quartal(4), 2)
        self.assertEqual(gasprognoseConstants.quartal(5), 2)
        self.assertEqual(gasprognoseConstants.quartal(6), 2)
        self.assertEqual(gasprognoseConstants.quartal(7), 3)
        self.assertEqual(gasprognoseConstants.quartal(8), 3)
        self.assertEqual(gasprognoseConstants.quartal(9), 3)
        self.assertEqual(gasprognoseConstants.quartal(10), 4)
        self.assertEqual(gasprognoseConstants.quartal(11), 4)
        self.assertEqual(gasprognoseConstants.quartal(12), 4)
        self.assertEqual(gasprognoseConstants.season(1), 2)
        self.assertEqual(gasprognoseConstants.season(2), 2)
        self.assertEqual(gasprognoseConstants.season(3), 2)
        self.assertEqual(gasprognoseConstants.season(4), 1)
        self.assertEqual(gasprognoseConstants.season(5), 1)
        self.assertEqual(gasprognoseConstants.season(6), 0)
        self.assertEqual(gasprognoseConstants.season(7), 0)
        self.assertEqual(gasprognoseConstants.season(8), 0)
        self.assertEqual(gasprognoseConstants.season(9), 1)
        self.assertEqual(gasprognoseConstants.season(10), 1)
        self.assertEqual(gasprognoseConstants.season(11), 2)
        self.assertEqual(gasprognoseConstants.season(12), 2)
        self.assertEqual(gasprognoseConstants.winter(1), 1)
        self.assertEqual(gasprognoseConstants.winter(2), 1)
        self.assertEqual(gasprognoseConstants.winter(3), 1)
        self.assertEqual(gasprognoseConstants.winter(4), 0)
        self.assertEqual(gasprognoseConstants.winter(5), 0)
        self.assertEqual(gasprognoseConstants.winter(6), 0)
        self.assertEqual(gasprognoseConstants.winter(7), 0)
        self.assertEqual(gasprognoseConstants.winter(8), 0)
        self.assertEqual(gasprognoseConstants.winter(9), 0)
        self.assertEqual(gasprognoseConstants.winter(10), 1)
        self.assertEqual(gasprognoseConstants.winter(11), 1)
        self.assertEqual(gasprognoseConstants.winter(12), 1)
        self.assertIsNone(gasprognoseConstants.winter(13))




if __name__ == '__main__':
    unittest.main()
