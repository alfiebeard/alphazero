import numpy as np
import random
import collections


def format_state(state):
    """Format a state for use in training"""
    state = np.array(state)[:, :, np.newaxis]
    state = state.astype(np.float32)
    return state


def format_replay(replay_buffer):
    """Format a series of replays for training"""
    states_ = np.zeros((len(replay_buffer), 3, 3, 1), dtype=np.float32)
    policy_ = np.zeros((len(replay_buffer), 9), dtype=np.float32)
    value_ = np.zeros((len(replay_buffer), 1), dtype=np.float32)

    for idx, event in enumerate(replay_buffer):
        states_[idx, :, :, :] = format_state(event[0])
        policy_[idx, :] = np.array(event[3])
        value_[idx, :] = event[4]

    return states_, policy_, value_


class Replay:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = collections.deque(maxlen=self.buffer_size)

    def add(self, example):
        self.buffer.extend(example)

    def full(self):
        if len(self.buffer) == self.buffer_size:
            return True
        else:
            return False

    def sample(self, batch_size):
        """Sample batch from replay buffer"""
        sample = random.sample(self.buffer, batch_size)

        return format_replay(sample)

    def sample_deduplicated(self, batch_size):
        """Sample batch from replay buffer and deduplicate overrepresented states"""
        states = []
        policy = []
        value = []

        for replay in self.buffer:
            try:
                # Duplicate - state already added to states -> append policy and value for averaging
                index = states.index(replay[0])     # Try find replay state in states
                policy[index].append(replay[3])
                value[index].append(replay[4])
            except ValueError:
                # No occurrence in states yet - .index fails and gives ValueError
                states.append(replay[0])
                policy.append([replay[3]])
                value.append([replay[4]])
        
        states_ = np.zeros((len(states), 3, 3, 1), dtype=np.float32)
        policy_ = np.zeros((len(states), 9), dtype=np.float32)
        value_ = np.zeros((len(states), 1), dtype=np.float32)

        # Average + format results
        for i in range(len(states)):
            states_[i, :, :, :] = format_state(states[i])
            policy_[i, :] = np.mean(np.array(policy[i]), axis=0)
            value_[i, :] = np.mean(np.array(value[i]))

        # Prevent negative sampling
        if batch_size > len(states):
            batch_size = len(states)

        # Sample
        samples = random.sample(range(len(states)), batch_size)

        return states_[samples], policy_[samples], value_[samples]

