import unittest
import numpy as np
import pandas as pd
import series_generator
import stop_loss


class TestSeriesGenerator(unittest.TestCase):

    def test_generate_date(self):
        start_date = "20100101"
        end_date = "20100102"

        dates = series_generator.generate_date(start_date, end_date, 1, False)
        self.assertEqual(len(dates), 2)

        dates = series_generator.generate_date(start_date, end_date, 1, True)
        self.assertEqual(len(dates), 1)

    def test_random_walk_generater(self):
        rw_series = series_generator.random_walk_generater(0, 1, "20100101", "20201231", 1, ignore_weekends = False)
        mean = rw_series.mean()
        std = rw_series.std()

        self.assertEqual(np.round(mean, 0), 0.0)
        self.assertEqual(np.round(std, 0), 1)

    def test_ar_1_return_generater(self):
        ar_1_series = series_generator.ar_1_return_generater(0, 1, 0.5, 0, "20100101", "20201231", ignore_weekends = False)
        mean = ar_1_series.mean()
        std = ar_1_series.std()

        self.assertEqual(np.round(mean, 0), 0.0)
        self.assertEqual(np.round(std, 0), 1)

    def test_regime_switching_return_generater(self):
        A = np.array([[1, 0], [1, 0]]) 
        rs_series, It = series_generator.regime_switching_return_generater(mean_1 = 0, std_1 = 1, mean_2 = 0, std_2 = 1, I0 = 1, trans_prob_matrix = A, 
                                               start_date = "20100101", end_date = "20201231", ignore_weekends = False)
        mean = rs_series.mean()
        std = rs_series.std()

        self.assertEqual(np.round(mean, 0), 0.0)
        self.assertEqual(np.round(std, 0), 1)

class TestStopLoss(unittest.TestCase):

    def test_simple_stop_loss_policy(self):
        start_date = "20100101"
        end_date = "20110101"

        dates = series_generator.generate_date(start_date, end_date, 1, False)
        constant_series = pd.Series(0.1, dates)
        constant_series.index = pd.to_datetime(constant_series.index)
        sl = stop_loss.StopLoss(constant_series, 0.0)

        st = sl.simple_stop_loss_policy(gamma = 0, delta = 0, J = 1, compounding = False)
        self.assertEqual(sum(st), len(constant_series))

        st = sl.simple_stop_loss_policy(gamma = -1, delta = 1, J = 1, compounding = False)
        self.assertEqual(sum(st), 1)

    def test_stopping_policy_performance(self):
        start_date = "20100101"
        end_date = "20110101"

        dates = series_generator.generate_date(start_date, end_date, 1, False)
        constant_series = pd.Series(0.1, dates)
        constant_series.index = pd.to_datetime(constant_series.index)
        sl = stop_loss.StopLoss(constant_series, 0.0)
        st = sl.simple_stop_loss_policy(gamma = 0, delta = 0, J = 1, compounding = False)

        with self.assertWarns(RuntimeWarning):
            performance, rst = sl.stopping_policy_performance(st)

        self.assertEqual(performance["Diff. b/w Expected Return"], 0)
        self.assertEqual(performance["Diff. b/w Standard Deviation"], 0)
        self.assertEqual(performance["Diff. b/w Sharpe Ratio"], 0)

if __name__ == '__main__':
    unittest.main()