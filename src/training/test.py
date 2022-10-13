from training.agent import LoadAgent
from training.mcts import MCTS
import numpy as np
import random


def play_one_game_agent_policy(env, agent, agent_player=1, with_mcts=True, render=False, strategy='random'):
    """Play one game with the agent vs a random policy or manually as a human and return the winner"""
    # If agent is a string - assume file path and load from there
    if isinstance(agent, str):
        agent = LoadAgent(load_path=agent)

    # Initialise environment
    obs, _ = env.reset()
    game_over = False

    # Initialise agent
    agent_start = False
    actions = []

    # Run game
    while not game_over:
        # If current player is agent - play with agent
        if env._player == agent_player:
            state = format_state(env, obs)
            if with_mcts:
                # If agent hasn't moved yet - load MCTS.
                if not agent_start:
                    mcts = MCTS(env, agent)
                    agent_start = True
                    
                # If agent_player is second mover - ignore first action, since it won't have explored this in MCTS.
                if agent_player == 1:
                    policy, action = mcts.run_test(actions=actions)
                else:
                    policy, action = mcts.run_test(actions=actions[1:])
            else:
                policy, _ = agent(state)
                policy = policy_remove_invalid_moves(policy, env.get_actions())
                action = np.argmax(policy)
        else:
            if strategy == 'random':
                # Select random action
                action = random.choice(env.get_actions())
            elif strategy == 'manual':
                action = int(input("Select action (" + ", ".join(map(str, env.get_actions())) + "): "))
            else:
                print("No such strategy, {0} exists".format(strategy))
                return

        # Only allow actions not already played - if not this affects the MCTS.
        if action not in actions:
            actions.append(action)

        # Execute action and get updated observations and game_over indicator
        obs, winner, game_over, _, _ = env.step(action)

        if render:
            env.render()

    return winner


def format_state(env, state):
    state = np.array(state)
    state = state.astype(np.float32)
    state = np.reshape(state, env._state_size)
    state = state[np.newaxis, ...]
    return state


def policy_remove_invalid_moves(policy, valid_moves):
    """Remove invalid actions from policy and return new policy with invalid moves set to 0"""
    if len(valid_moves) == 1:
        # If only one valid move - take it.
        return [1 if i in valid_moves else 0 for i in range(len(policy))]
    else:
        for i in range(len(policy)):
            if i not in valid_moves:
                policy[i] = 0
        
        policy_sum = sum(policy)
        return [policy_i / policy_sum for policy_i in policy]


def random_test(env, agent, num_games=100, agent_player=1, with_mcts=True):
    """Run a batch of simulations and return statistics for agent_player"""
    # agent_player = 1, means the agent moves first, agent_player = -1, means the agent moves second.
    # with_mcts = True, means MCTS is used for testing the agent
    # If agent is a string - assume file path and load from there
    if isinstance(agent, str):
        agent = LoadAgent(load_path=agent)

    # Run games and collect winners
    winners = []
    for _ in range(num_games):
        winners.append(play_one_game_agent_policy(env, agent, agent_player=agent_player, with_mcts=with_mcts, strategy='random'))

    # Calculate statistics
    win_percent = sum([abs(w) for w in winners if w == agent_player]) / num_games
    draw_percent = sum([1.0 for w in winners if w == 0]) / num_games
    loss_percent = sum([abs(w) for w in winners if w == -agent_player]) / num_games
    
    print("Win percentage: {0}%".format(100 * win_percent))
    print("Draw percentage: {0}%".format(100 * draw_percent))
    print("Loss percentage: {0}%".format(100 * loss_percent))
    
    return {"win_percent": win_percent, "draw_percent": draw_percent, "loss_percent": loss_percent}

