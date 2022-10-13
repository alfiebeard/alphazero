import numpy as np
import random


class Node:
    def __init__(self, prior_p, player, parent=None):
        self.parent = parent
        self.visit_count = 0    # N - number of visits
        self.total_value = 0    # W - total value
        self.mean_value = 0     # Q - mean value
        self.prior_p = prior_p  # P - probability
        self.player = player    # Player at current node
        self.children = {}
        self.expanded = False

    def is_expanded(self):
        if len(self.children) > 0:
            return True
        else:
            return False


class MCTS:
    # MCTS algorithm
    def __init__(self, env, agent, num_simulations=25, dirichlet_alpha=0.7, noise_fraction=0.7, c_base=1,
                 c_init=0, num_sampling_moves=6):
        self.env = env
        self.env_clone = None
        self.all_actions = env.get_actions()      # Assuming you can make all possible moves at the start
        self.agent = agent
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_fraction = noise_fraction
        self.c_base = c_base
        self.c_init = c_init
        self.num_sampling_moves = num_sampling_moves
        self.root = Node(0, self.env._player)
        self.current_node = self.root

    def run(self):
        """Run one step using MCTS returning an action and policy"""
        self.add_exploration_noise(self.current_node)    # Introduce some noise to encourage exploration

        # Run MCTS simulations
        for _ in range(self.num_simulations):
            self.env_clone = self.env.clone()
            self.search(self.current_node)

        # Set temerpature to encourage shift from exploration to exploitation further into a game.
        tau = 1
        if len(self.env._history) > self.num_sampling_moves:
            # As time increases, tau tends to 0 - this exploits more than explores.
            tau = 0.2

        # Return policy vector, which is the number of visits, as a dict, {'action': # visits, ...}
        policy_sum = sum([node_info.visit_count ** (1 / tau) for _, node_info in self.current_node.children.items()])
        policy = [(self.current_node.children[action].visit_count ** (1 / tau)) / policy_sum if action in self.current_node.children else 0 for
                  action in self.all_actions]

        action = self.select_action(self.current_node)

        # Update current_node to reflect action
        self.current_node = self.current_node.children[action]

        return policy, action

    def run_test(self, actions=[]):
        """Run one step using MCTS at test time (i.e., full exploitation) and return action and policy"""
        # Move to current node by moving through list of actions.
        self.current_node = self.root
        for action in actions:
            self.current_node = self.current_node.children[action]

        # Run MCTS simulations
        for _ in range(self.num_simulations):
            self.env_clone = self.env.clone()
            self.search(self.current_node)

        # Return policy vector, which is the number of visits, as a dict, {'action': # visits, ...}
        tau = 0.01  # Set temperature to exploit
        policy_sum = sum([node_info.visit_count ** (1 / tau) for _, node_info in self.current_node.children.items()])
        policy = [(self.current_node.children[action].visit_count ** (1 / tau)) / policy_sum if action in self.current_node.children else 0 for
                  action in self.all_actions]

        action = self.select_action(self.current_node, explore=False)

        return policy, action

    def search(self, root):
        """Execute one MCTS search from root node"""

        # Select a leaf node - traverse the tree from the root until in a leaf.
        leaf, search_path = self.select_leaf(root)

        # Expand leaf node - if the leaf's state is not terminal, expand the leaf.
        if not self.env_clone._terminal:
            value = self.expand(leaf)
        else:
            value = self.env_clone.get_result(leaf.player)

        # Backpropagate count and value up tree
        self.backpropagate(search_path, value)

    def select_leaf(self, node):
        """Traverse the tree until in an unexpanded node and return this node"""

        # search_path for storing the nodes visited in.
        search_path = [node]

        # Traverse tree until in leaf node
        while node.expanded:
            action, node = self.select_child(node)  # Select the best child
            self.env_clone.step(action)
            search_path.append(node)

        return node, search_path

    def select_child(self, node):
        """Select the best child from current node based on UCB score"""
        max_ucb = None
        selected_child = None
        selected_action = None

        # Iterate through all children selecting child with max ucb_score.
        for action, child in node.children.items():
            score = self.ucb_score(node, child)
            # If no max from before or score is larger than max, update max score and child.
            if max_ucb is None or max_ucb < score:
                max_ucb = score
                selected_child = child
                selected_action = action
        return selected_action, selected_child

    def ucb_score(self, parent, child):
        """Calculate UCB score for a node"""
        c = np.log((parent.visit_count + self.c_base + 1) / self.c_base) + self.c_init
        u = c * child.prior_p * np.sqrt(parent.visit_count) / (child.visit_count + 1)

        q = - child.mean_value  # Negative since value is for the parent player, which is the other player
        return q + u

    def expand(self, node):
        """Expand a node using the neural network and predict value of it's state"""
        # Get the action probabilities (a dictionary of actions and probabilities) and value of the state.
        states = np.array([self.env_clone.get_observation(node.player)])
        states = states.astype(np.float32)
        states = np.reshape(states, self.env._state_size)
        states = states[np.newaxis, ...]

        # Predict policy and value with agent
        probs, value = self.agent(states)

        # Get policy from possible actions and probabilities
        policy = {a: probs[a] for a in self.env_clone.get_actions()}
        policy_sum = sum(policy.values())

        # Create child nodes
        if len(policy) == 1:
            # If there is one action in policy - set prior to 1, as only option - this prevents divide by zero errors.
            for action in policy.keys():
                node.children[action] = Node(1, player=node.player*-1)
        else:
            for action, p in policy.items():
                node.children[action] = Node(p / policy_sum, player=node.player*-1)    # Add prior_p to all children
        node.expanded = True

        return value

    def backpropagate(self, search_path, value):
        """Backpropagate visit count and value up the tree"""
        node_value_player = search_path[-1].player      # Node player is the player for the current leaf node
        for node in search_path:
            node.visit_count += 1
            node.total_value += value if node.player == node_value_player else -value     # Add value if node player same as current player, otherwise subtract
            node.mean_value = node.total_value / node.visit_count

    @staticmethod
    def select_action(root, explore=True):
        """Return the "best" action based on the MCTS"""

        # Get the visit counts
        visit_counts = [(child.visit_count, action) for action, child in root.children.items()]

        if explore:
            visit_vector = [visits[0] for visits in visit_counts]
            action_vector = [actions[1] for actions in visit_counts]
            action = np.random.choice(action_vector, p=[x / sum(visit_vector) for x in visit_vector])
        else:
            _, action = max(visit_counts)

        return action

    def add_exploration_noise(self, node):
        """Add exploration noise to probabilities in a node's children using dirichlet noise"""
        actions = node.children.keys()
        noise = np.random.gamma(self.dirichlet_alpha, 1, len(actions))
        noise_sum = sum(noise)
        normalised_noise = [n / noise_sum for n in noise]
        for a, n in zip(actions, normalised_noise):
            node.children[a].prior_p = node.children[a].prior_p * (1 - self.noise_fraction) + n * self.noise_fraction

