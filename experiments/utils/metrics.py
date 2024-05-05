import pandas as pd
from lcs.metrics import population_metrics


def common_metrics(agent, env):
    metrics = {}

    pop = agent.get_population()
    agent_name = agent.__class__.__name__

    if hasattr(agent, 'rho'):
        metrics['rho'] = agent.rho
        agent_name += "_v" + agent.cfg.rho_update_version
    else:
        metrics['rho'] = 0

    metrics['agent'] = agent_name
    metrics['reliable'] = len([cl for cl in pop if cl.is_reliable()])

    metrics.update(population_metrics(pop, env))

    return metrics


def parse_metrics(metrics):
    # idx = [d['agent'] for d in metrics]

    data = [[
        d['agent'],
        d['trial'],
        d['steps_in_trial'],
        d['rho'],
        d['population'],
        d['reliable']] for d in metrics]

    df = pd.DataFrame(
        data,
        columns=['agent', 'trial', 'steps_in_trial', 'rho', 'population', 'reliable'],
        # index=idx
    )

    df['phase'] = df.trial.map(
        lambda t: "explore" if t % 2 == 0 else "exploit")

    return df
