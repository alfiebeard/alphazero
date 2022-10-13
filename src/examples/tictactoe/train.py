from tictactoe_gym.envs.tictactoe_env import TicTacToeEnv
from training.train import run
from examples.tictactoe.neural_net import TicTacToeNeuralNet
from training.agent import Agent
import argparse


def train(env, agent, checkpoint_path="src/examples/tictactoe/checkpoints", save_path="src/examples/tictactoe/saved_models/tictactoe_agent", num_games=100, epochs=2, batch_size=300, replay_buffer_size=600):
    run(env=env, agent=agent, save_path=save_path, checkpoint_path=checkpoint_path, num_games=num_games, epochs=epochs, batch_size=batch_size, replay_buffer_size=replay_buffer_size)


# Training script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default="src/examples/tictactoe/checkpoints", help="Checkpoints for saving agent progress")
    parser.add_argument('--save_path', type=str, default="src/examples/tictactoe/saved_models/tictactoe_agent", help="Checkpoints for saving agent progress")
    parser.add_argument('--num_games', type=int, default=500, help="Number of test games to run")
    parser.add_argument('--epochs', type=int, default=2, help="Number of epochs in training")
    parser.add_argument('--batch_size', type=int, default=300, help="Batch size to sample")
    parser.add_argument('--replay_buffer_size', type=int, default=600, help="Replay buffer size")
    args = parser.parse_args()

    env = TicTacToeEnv()
    neural_net = TicTacToeNeuralNet()
    agent = Agent(neural_net)

    train(env=env, agent=agent, checkpoint_path=args.checkpoint_path, save_path=args.save_path, num_games=args.num_games, epochs=args.epochs, batch_size=args.batch_size, replay_buffer_size=args.replay_buffer_size)

