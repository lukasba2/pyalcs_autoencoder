import logging
from collections import defaultdict

import lcs.agents.aacs2 as aacs2
import lcs.agents.acs2 as acs2
import pandas as pd
from utils.metrics import parse_metrics


def avg_experiments(n, fun, envs: dict, trials, params):
    results = defaultdict(dict)

    for name, env in envs.items():
        pops_acs2 = []
        pops_aacs2v1 = []
        pops_aacs2v2 = []
        dfs = []

        for i in range(n):
            # Call the experiment function and store the result
            results_env = fun(env=env, trials=trials, params=params)

            # Extract the results
            p_acs2, p_aacs2v1, p_aacs2v2, m = results_env['pops_acs2'], results_env['pops_aacs2v1'], results_env['pops_aacs2v2'], results_env['m']

            pops_acs2.append(p_acs2)
            pops_aacs2v1.append(p_aacs2v1)
            pops_aacs2v2.append(p_aacs2v2)

            df = parse_metrics(m)
            dfs.append(df)

        all_dfs = pd.concat(dfs)
        agg_df = all_dfs.groupby(['agent', 'trial', 'phase']).mean().reset_index(
            level='phase')

        results[name]["acs2"] = pops_acs2
        results[name]["aacs2v1"] = pops_aacs2v1
        results[name]["aacs2v2"] = pops_aacs2v2
        results[name]["agg_df"] = agg_df

        results["all_dfs"] = all_dfs

    return results


def run_experiments_alternating(env, trials, params):
    logging.info('Starting ACS2 experiments')
    acs2_cfg = acs2.Configuration(
        classifier_length=params['perception_bits'],
        number_of_possible_actions=params['possible_actions'],
        do_ga=params['do_ga'],
        beta=params['beta'],
        epsilon=params['epsilon'],
        gamma=params['gamma'],
        user_metrics_collector_fcn=params['user_metrics_collector_fcn'],
        biased_exploration_prob=params['biased_exploration_prob'],
        metrics_trial_frequency=params['metrics_trial_freq'])

    acs2_agent = acs2.ACS2(acs2_cfg)
    pop_acs2, metrics_acs2 = acs2_agent.explore_exploit(env, trials)

    logging.info('Starting AACS2-v1 experiments')
    aacs2v1_cfg = aacs2.Configuration(
        classifier_length=params['perception_bits'],
        number_of_possible_actions=params['possible_actions'],
        do_ga=params['do_ga'],
        beta=params['beta'],
        epsilon=params['epsilon'],
        gamma=params['gamma'],
        zeta=params['zeta'],
        rho_update_version='1',
        user_metrics_collector_fcn=params['user_metrics_collector_fcn'],
        biased_exploration_prob=params['biased_exploration_prob'],
        metrics_trial_frequency=params['metrics_trial_freq'])

    aacs2v1_agent = aacs2.AACS2(aacs2v1_cfg)
    pop_aacs2v1, metrics_aacs2v1 = aacs2v1_agent.explore_exploit(env, trials)

    logging.info('Starting AACS2-v2 experiments')
    aacs2v2_cfg = aacs2.Configuration(
        classifier_length=params['perception_bits'],
        number_of_possible_actions=params['possible_actions'],
        do_ga=params['do_ga'],
        beta=params['beta'],
        epsilon=params['epsilon'],
        gamma=params['gamma'],
        zeta=params['zeta'],
        rho_update_version='2',
        user_metrics_collector_fcn=params['user_metrics_collector_fcn'],
        biased_exploration_prob=params['biased_exploration_prob'],
        metrics_trial_frequency=params['metrics_trial_freq'])

    aacs2v2_agent = aacs2.AACS2(aacs2v2_cfg)
    pop_aacs2v2, metrics_aacs2v2 = aacs2v2_agent.explore_exploit(env, trials)

    m = []
    m.extend(metrics_acs2)
    m.extend(metrics_aacs2v1)
    m.extend(metrics_aacs2v2)

    return pop_acs2, pop_aacs2v1, pop_aacs2v2, m
