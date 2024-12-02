from visualize import (
    visualize_portfolio_net_worth,
    visualize_multiple_portfolio_net_worth,
)


def test_agent(env, agent, stock_data, n_tests=1000, visualize=False):
    """
    Test a single agent and track performance metrics, with an option to visualize the results.

    Parameters:
    - env: The trading environment.
    - agent: The agent to be tested.
    - stock_data: Data for the stocks in the environment.
    - n_tests: Number of tests to run (default: 1000).
    - visualize: Boolean flag to enable or disable visualization (default: False).

    Returns:
    - A dictionary containing steps, balances, net worths, and shares held.
    """
    # Initialize metrics tracking
    metrics = {
        "steps": [],
        "balances": [],
        "net_worths": [],
        "shares_held": {ticker: [] for ticker in stock_data.keys()},
    }

    # Reset the environment before starting the tests
    obs = env.reset()

    for i in range(n_tests):
        metrics["steps"].append(i)
        action = agent.predict(obs)
        obs, rewards, dones, infos = env.step(action)
        if visualize:
            env.render()

        # Track metrics
        metrics["balances"].append(env.get_attr("balance")[0])
        metrics["net_worths"].append(env.get_attr("net_worth")[0])
        env_shares_held = env.get_attr("shares_held")[0]

        # Update shares held for each ticker
        for ticker in stock_data.keys():
            if ticker in env_shares_held:
                metrics["shares_held"][ticker].append(env_shares_held[ticker])
            else:
                metrics["shares_held"][ticker].append(
                    0
                )  # Append 0 if ticker is not found

        if dones:
            obs = env.reset()

    return metrics


def test_and_visualize_agents(env, agents, training_data, n_tests=1000):
    metrics = {}
    for agent_name, agent in agents.items():
        print(f"Testing {agent_name}...")
        metrics[agent_name] = test_agent(
            env, agent, training_data, n_tests=n_tests, visualize=True
        )
        print(f"Done testing {agent_name}!")

    print("-" * 50)
    print("All agents tested!")
    print("-" * 50)

    # Extract net worths for visualization
    net_worths = [metrics[agent_name]["net_worths"] for agent_name in agents.keys()]
    steps = next(iter(metrics.values()))[
        "steps"
    ]  # Assuming all agents have the same step count for simplicity

    # Visualize the performance metrics of multiple agents
    visualize_multiple_portfolio_net_worth(steps, net_worths, list(agents.keys()))
