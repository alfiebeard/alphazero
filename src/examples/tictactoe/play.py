from tictactoe_gym.envs.tictactoe_env import TicTacToeEnv
from training.test import play_one_game_agent_policy
import argparse


def play(env=TicTacToeEnv(), agent='src/examples/tictactoe/saved_models/tictactoe_agent_trained' , agent_player=1, with_mcts=True, render=True, strategy='manual'):
    winner = play_one_game_agent_policy(env, agent=agent, agent_player=agent_player, with_mcts=with_mcts, render=render, strategy=strategy)

    if winner == agent_player:
        print("The agent won!")
    elif winner == -agent_player:
        print("Congratulations you won!")
    else:
        print("It's a draw!")


# Play against trained agent script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default="src/examples/tictactoe/saved_models/tictactoe_agent_trained", help="Filepath to an agent")
    parser.add_argument('--agent_player', type=int, default=1, help="1 if the agent moves first, -1 if the agent moves second")
    parser.add_argument('--with_mcts', type=bool, default=True, help="Use Monte Carlo Tree Search when running")
    parser.add_argument('--render', type=bool, default=True, help="Render results to command line")
    parser.add_argument('--strategy', type=str, default="manual", help="manual or random, depending on if you want to play")
    args = parser.parse_args()

    play(env=TicTacToeEnv(), agent=args.agent , agent_player=args.agent_player, with_mcts=args.with_mcts, render=args.render, strategy=args.strategy)

