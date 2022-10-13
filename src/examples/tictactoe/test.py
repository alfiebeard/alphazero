from tictactoe_gym.envs.tictactoe_env import TicTacToeEnv
from training.test import random_test
import argparse


def test(env=TicTacToeEnv(), agent='src/examples/tictactoe/saved_models/tictactoe_agent_trained', agent_player=1, num_games=100):
    random_test(env, agent=agent, agent_player=agent_player, num_games=num_games)


# Testing script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default="src/examples/tictactoe/saved_models/tictactoe_agent_trained", help="Filepath to an agent")
    parser.add_argument('--agent_player', type=int, default=1, help="1 if the agent moves first, -1 if the agent moves second")
    parser.add_argument('--num_games', type=int, default=100, help="Number of test games to run")
    args = parser.parse_args()

    test(env=TicTacToeEnv(), agent=args.agent, agent_player=args.agent_player, num_games=args.num_games)

