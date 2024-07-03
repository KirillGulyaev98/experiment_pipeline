import pandas as pd
import numpy as np
import abc
import config as cfg
from itertools import product
from metric_builder import Metric, CalculateMetric
from stattests import TTestFromStats, MannWhitneyU, ProportionTest, calculate_statistics_ttest, calculate_linearization, calculate_statistics_utest, calculate_statistics_proportion_test


class Report:
    def __init__(self, report):
        self.report = report


class BuildMetricReport_ttest:
    def __call__(self, calculated_metric, metric_items) -> Report:
        ttest = TTestFromStats()
        cfg.logger.info(f"{metric_items.name}")

        df_ = calculate_linearization(calculated_metric)
        stats = calculate_statistics_ttest(df_, metric_items.type)
        criteria_res = ttest(stats)

        report_items = pd.DataFrame({
            "metric_name": metric_items.name,
            "mean_0": stats.mean_0,
            "mean_1": stats.mean_1,
            "var_0": stats.var_0,
            "var_1": stats.var_1,
            "delta": stats.mean_1 - stats.mean_0,
            "lift":  (stats.mean_1 - stats.mean_0) / stats.mean_0,
            "pvalue": criteria_res.pvalue,
            "statistic": criteria_res.statistic
        }, index=[0])

        return Report(report_items)

class BuildMetricReport_utest:
    def __call__(self, calculated_metric, metric_items) -> Report:
        mannwhitney = MannWhitneyU()
        cfg.logger.info(f"{metric_items.name}")

        df_ = calculated_metric  # Assuming no linearization needed for Mann-Whitney
        stats = calculate_statistics_utest(df_, metric_items.type)
        criteria_res = mannwhitney(stats)

        report_items = pd.DataFrame({
            "metric_name": metric_items.name,
            "delta": np.median(stats.values_1) - np.median(stats.values_0),  # Medians for Mann-Whitney
            "pvalue": criteria_res.pvalue,
            "statistic": criteria_res.statistic
        }, index=[0])

        return Report(report_items)

class BuildMetricReport_proportion_test:
    def __call__(self, calculated_metric, metric_items) -> Report:
        proportion_test = ProportionTest()
        cfg.logger.info(f"{metric_items.name}")

        df_ = calculated_metric
        stats = calculate_statistics_proportion_test(df_, metric_items.type)
        criteria_res = proportion_test(stats)

        report_items = pd.DataFrame({
            "metric_name": metric_items.name,
            "count_0": stats.count_0,
            "count_1": stats.count_1,
            "nobs_0": stats.nobs_0,
            "nobs_1": stats.nobs_1,
            "delta": (stats.count_1 / stats.nobs_1) - (stats.count_0 / stats.nobs_0),
            "lift": ((stats.count_1 / stats.nobs_1) - (stats.count_0 / stats.nobs_0)) / (stats.count_0 / stats.nobs_0),
            "pvalue": criteria_res.pvalue,
            "statistic": criteria_res.statistic
        }, index=[0])

        return Report(report_items)


def build_experiment_report_ttest(df, metric_config):
    build_metric_report = BuildMetricReport_ttest()
    reports = []

    for metric_params in metric_config:
        metric_parsed = Metric(metric_params)
        calculated_metric = CalculateMetric(metric_parsed)(df)
        metric_report = build_metric_report(calculated_metric, metric_parsed)
        reports.append(metric_report.report)

    return pd.concat(reports)

def build_experiment_report_utest(df, metric_config):
    build_metric_report = BuildMetricReport_utest()
    reports = []

    for metric_params in metric_config:
        metric_parsed = Metric(metric_params)
        calculated_metric = CalculateMetric(metric_parsed)(df)
        metric_report = build_metric_report(calculated_metric, metric_parsed)
        reports.append(metric_report.report)

    return pd.concat(reports)

def build_experiment_report_proportion_test(df, metric_config):
    build_metric_report = BuildMetricReport_proportion_test()
    reports = []

    for metric_params in metric_config:
        metric_parsed = Metric(metric_params)
        calculated_metric = CalculateMetric(metric_parsed)(df)
        metric_report = build_metric_report(calculated_metric, metric_parsed)
        reports.append(metric_report.report)

    return pd.concat(reports)

