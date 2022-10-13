from training.mcts import MCTS
from training.test import random_test
from training.replay import Replay
import numpy as np
import matplotlib.pyplot as plt
import time
import collections
import tensorflow as tf


def run_game(env, agent):
    """Run one game of the env with the current agent using MCTS and return the game history"""
    # Initialise env and search
    env.reset()
    game_over = False
    game_history = []
    reward = 0
    search = MCTS(env, agent)

    # Play game
    while not game_over:
        policy, action = search.run()   # Run MCTS search to get policy and action
        game_history.append([env.get_observation(env._player), action, env._player, policy])     # Store in game history
        _, reward, game_over, _, _ = env.step(action)    # Execute action

    # Add game reward for each player into game history - win +1, draw 0, loss -1, depends on who the player is, so multiply.
    for idx, event in enumerate(game_history):
        event.append(reward * event[2])
        event.append(idx)   # Add index of game step - can be useful to track
        game_history[idx] = event

    return game_history
            

def run(env, agent, save_path, checkpoint_path, num_games=1000, epochs=2, batch_size=750, replay_buffer_size=1500):
    """Run training of an agent on a gym environment and save the agent"""
    # Initialise replay_buffer
    replay = Replay(buffer_size=replay_buffer_size)
    total_training_examples = 0

    # Initialise checkpoints
    ckpt = tf.train.Checkpoint(agent.neural_net)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    ckpt_save_interval = 50000  # Save checkpoint every ckpt_save_interval batches

    # Test progress of agent every agent_progress_interval steps
    agent_progress_interval = 10
    progress_interval = np.floor(num_games / agent_progress_interval)
    test_results = []

    # Run simulations
    start_time = time.time()
    for i in range(num_games):
        # If testing agent progress
        if i % progress_interval == 0:
            # Run agent_progress_interval test games with agent and track percentage of games lost
            print("Progress at {0}%".format(round(i * 100 / num_games, 2)))
            print("Testing")
            results = random_test(env, agent, with_mcts=False)
            test_results.append(results['win_percent'])
            env.reset()

        # Play one game using the agent and MCTS and store history in replay_buffer
        replay.add(run_game(env, agent))

        # If replay_buffer full start training the agent
        if replay.full():
            # Sample from replay buffer - removing duplicates to handle overreprensentation of some states.
            states_batch, policy_batch, value_batch = replay.sample_deduplicated(batch_size=batch_size)
            # Train agent for a set number of epochs on batches
            for _ in range(epochs):
                agent.train_agent_batch(states_batch, policy_batch, value_batch)
                total_training_examples += batch_size
            
            # Save checkpoint every ckpt_save_interval batches
            if total_training_examples % ckpt_save_interval == 0:
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint after {0} samples at {1}'.format(total_training_examples, ckpt_save_path))
                print('Loss {0}'.format(agent.loss_values[-1]))
        
        env.reset()

    agent.save(save_path)

    # Print training time
    print('Runtime {0}s'.format(time.time() - start_time))

    # Plot training loss over epochs
    plt.subplot(1, 2, 1)
    plt.plot(list(range(len(agent.loss_values))), agent.loss_values, label="Total Loss", color="red")
    plt.plot(list(range(len(agent.policy_loss_values))), agent.policy_loss_values, label="Policy Loss", color="blue")
    plt.plot(list(range(len(agent.value_loss_values))), agent.value_loss_values, label="Value Loss", color="green")
    plt.title("Training loss")
    plt.xlabel("Training steps")
    plt.ylabel("Loss")
    plt.legend()

    # Plot the percentage of losses in testing over training duration
    plt.subplot(1, 2, 2)
    plt.plot(list(range(len(test_results))), [t * 100 for t in test_results], label="Test results", color="blue")
    plt.title("Win percentage over training")
    plt.xlabel("Training progress (%)")
    plt.ylabel("Win percentage (%)")
    plt.legend()

    plt.show()

