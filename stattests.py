import pandas as pd
import numpy as np
import abc

from scipy.stats import ttest_ind_from_stats, ttest_ind, mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest

import config as cfg


class EstimatorCriteriaValues:
    def __init__(self, pvalue: float, statistic: float):
        self.pvalue = pvalue
        self.statistic = statistic


class Statistics_ttest:
    def __init__(self, mean_0: float, mean_1: float, var_0: float, var_1: float, n_0: int, n_1: int):
        self.mean_0 = mean_0
        self.mean_1 = mean_1
        self.var_0 = var_0
        self.var_1 = var_1
        self.n_0 = n_0
        self.n_1 = n_1

class Statistics_proportion_test:
    def __init__(self, count_0: int, count_1: int, nobs_0: int, nobs_1: int):
        self.count_0 = count_0
        self.count_1 = count_1
        self.nobs_0 = nobs_0
        self.nobs_1 = nobs_1

class Statistics_utest:
    def __init__(self, values_0: np.ndarray, values_1: np.ndarray):
        self.values_0 = values_0
        self.values_1 = values_1


class MetricStats_ttest(abc.ABC):
    @abc.abstractmethod
    def __call__(self, df) -> Statistics_ttest:
        pass

class MetricStats_proportion_test(abc.ABC):
    @abc.abstractmethod
    def __call__(self, df) -> Statistics_proportion_test:
        pass

class MetricStats_utest(abc.ABC):
    @abc.abstractmethod
    def __call__(self, df) -> Statistics_utest:
        pass

class Estimator_ttest(abc.ABC):
    @abc.abstractmethod
    def __call__(self, Statistics_ttest) -> EstimatorCriteriaValues:
        pass

class Estimator_proportion(abc.ABC):
    @abc.abstractmethod
    def __call__(self, Statistics_proportion_test) -> EstimatorCriteriaValues:
        pass

class Estimator_utest(abc.ABC):
    @abc.abstractmethod
    def __call__(self, Statistics_utest) -> EstimatorCriteriaValues:
        pass

class BaseStatsRatio(MetricStats_ttest):
    def __call__(self, df) -> Statistics_ttest:
        _unique_variants = df[cfg.VARIANT_COL].unique()
        n_0 = sum(df['n'][df[cfg.VARIANT_COL] == _unique_variants[0]])
        n_1 = sum(df['n'][df[cfg.VARIANT_COL] == _unique_variants[1]])
        mean_0 = sum(df['num'][df[cfg.VARIANT_COL] == _unique_variants[0]]) / sum(df['den'][df[cfg.VARIANT_COL] == _unique_variants[0]])
        mean_1 = sum(df['num'][df[cfg.VARIANT_COL] == _unique_variants[1]]) / sum(df['den'][df[cfg.VARIANT_COL] == _unique_variants[1]])
        var_0 = df['l_ratio'][df[cfg.VARIANT_COL] == _unique_variants[0]].var()
        var_1 = df['l_ratio'][df[cfg.VARIANT_COL] == _unique_variants[1]].var()

        return Statistics_ttest(mean_0, mean_1, var_0, var_1, n_0, n_1)

class BaseStatsProportionTest(MetricStats_proportion_test):
    def __call__(self, df) -> Statistics_proportion_test:
        _unique_variants = df[cfg.VARIANT_COL].unique()
        count_0 = sum(df['num'][df[cfg.VARIANT_COL] == _unique_variants[0]])
        count_1 = sum(df['num'][df[cfg.VARIANT_COL] == _unique_variants[1]])
        nobs_0 = sum(df['den'][df[cfg.VARIANT_COL] == _unique_variants[0]])
        nobs_1 = sum(df['den'][df[cfg.VARIANT_COL] == _unique_variants[1]])

        return Statistics_proportion_test(count_0, count_1, nobs_0, nobs_1)

class BaseStatsMannWhitney(MetricStats_utest):
    def __call__(self, df) -> Statistics_utest:
        _unique_variants = df[cfg.VARIANT_COL].unique()
        values_0 = (df['num'][df[cfg.VARIANT_COL] == _unique_variants[0]] / df['den'][df[cfg.VARIANT_COL] == _unique_variants[0]]).values
        values_1 = (df['num'][df[cfg.VARIANT_COL] == _unique_variants[1]] / df['den'][df[cfg.VARIANT_COL] == _unique_variants[1]]).values

        return Statistics_utest(values_0, values_1)

class MannWhitneyU(Estimator_utest):
    def __call__(self, stat: Statistics_utest) -> EstimatorCriteriaValues:
        try:
            statistic, pvalue = mannwhitneyu(stat.values_0, stat.values_1, alternative='two-sided')
        except Exception as e:
            cfg.logger.error(e)
            statistic, pvalue = None, None

        return EstimatorCriteriaValues(pvalue, statistic)

class Linearization():

    def __call__(self, num_0, den_0, num_1, den_1):
        k = np.sum(num_0) / np.sum(den_0)
        l_0 = num_0 - k * den_0
        l_1 = num_1 - k * den_1
        return l_0, l_1


class TTestFromStats(Estimator_ttest):

    def __call__(self, stat: Statistics_ttest) -> EstimatorCriteriaValues:
        try:
            statistic, pvalue = ttest_ind_from_stats(
                mean1=stat.mean_0,
                std1=np.sqrt(stat.var_0),
                nobs1=stat.n_0,
                mean2=stat.mean_1,
                std2=np.sqrt(stat.var_1),
                nobs2=stat.n_1
            )
        except Exception as e:
            cfg.logger.error(e)
            statistic, pvalue = None, None

        return EstimatorCriteriaValues(pvalue, statistic)

class ProportionTest(Estimator_utest):
    def __call__(self, stat: Statistics_proportion_test) -> EstimatorCriteriaValues:
        try:
            if stat.nobs_0 == 0 or stat.nobs_1 == 0:
                raise ValueError("Number of observations (nobs) should not be zero")
            statistic, pvalue = proportions_ztest(
                [stat.count_0, stat.count_1],
                [stat.nobs_0, stat.nobs_1]
            )
        except Exception as e:
            cfg.logger.error(e)
            statistic, pvalue = None, None

        return EstimatorCriteriaValues(pvalue, statistic)


def calculate_statistics_ttest(df, type):
    mappings = {
        "ratio": BaseStatsRatio()
        # TODO расчет статистик не для ratio
    }

    calculate_method = mappings[type]

    return calculate_method(df)


def calculate_statistics_utest(df, type):
    mappings = {
        "ratio": BaseStatsMannWhitney()
    }
    calculate_method = mappings[type]

    return calculate_method(df)

def calculate_statistics_proportion_test(df, type):
    mappings = {
        "ratio": BaseStatsProportionTest()
    }
    calculate_method = mappings[type]

    return calculate_method(df)


def calculate_linearization(df):
    _variants = df[cfg.VARIANT_COL].unique()
    linearization = Linearization()

    df['l_ratio'] = 0
    if (df['den'] == df['n']).all():
        df.loc[df[cfg.VARIANT_COL] == _variants[0], 'l_ratio'] = df.loc[df[cfg.VARIANT_COL] == _variants[0], 'num']
        df.loc[df[cfg.VARIANT_COL] == _variants[1], 'l_ratio'] = df.loc[df[cfg.VARIANT_COL] == _variants[1], 'num']
    else:
        l_0, l_1 = linearization(
            df['num'][df[cfg.VARIANT_COL] == _variants[0]],
            df['den'][df[cfg.VARIANT_COL] == _variants[0]],
            df['num'][df[cfg.VARIANT_COL] == _variants[1]],
            df['den'][df[cfg.VARIANT_COL] == _variants[1]]
        )
        df.loc[df[cfg.VARIANT_COL] == _variants[0], 'l_ratio'] = l_0
        df.loc[df[cfg.VARIANT_COL] == _variants[1], 'l_ratio'] = l_1

    return df

