from agents import create_env_and_train_agents, compare_and_plot_agents
from tests import test_and_visualize_agents, test_agent
from data import Data


if __name__ == "__main__":
    data = Data()
    data.load_data()
    training_data = data.training()

    total_timesteps = 10000
    env, ppo_agent, a2c_agent, ddpg_agent, ensemble_agent = create_env_and_train_agents(
        training_data, total_timesteps
    )

    n_tests = 1000
    agents = {
        "PPO Agent": ppo_agent,
        "A2C Agent": a2c_agent,
        "DDPG Agent": ddpg_agent,
        "Ensemble Agent": ensemble_agent,
    }
    test_and_visualize_agents(env, agents, training_data, n_tests=n_tests)

    agents_metrics = [
        test_agent(env, agent, training_data, n_tests=n_tests, visualize=False)
        for agent in agents.values()
    ]
    compare_and_plot_agents(agents_metrics, list(agents.keys()))
