import tensorflow as tf


class Agent:
    def __init__(self, neural_net):
        # Create an instance of the neural network
        self.neural_net = neural_net

        self.policy_loss_values = []
        self.value_loss_values = []
        self.loss_values = []

    def __call__(self, states):
        return self.predict(states)

    def predict(self, states):
        p, v = self.neural_net(states)
        return p.numpy()[0], v.numpy()[0][0]

    @tf.function(input_signature=[
            tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        ]
    )
    def train_agent(self, state, target_policy, target_value):
        loss, policy_loss, value_loss = self.neural_net.loss(state, target_policy, target_value, training=True)

        self.neural_net.policy_loss_metric(policy_loss)
        self.neural_net.value_loss_metric(value_loss)
        self.neural_net.loss_metric(loss)

    def train_agent_batch(self, state, target_policy, target_value):
        self.neural_net.policy_loss_metric.reset_state()
        self.neural_net.value_loss_metric.reset_state()
        self.neural_net.loss_metric.reset_state()

        self.train_agent(state, target_policy, target_value)

        self.policy_loss_values.append(self.neural_net.policy_loss_metric.result().numpy())
        self.value_loss_values.append(self.neural_net.value_loss_metric.result().numpy())
        self.loss_values.append(self.neural_net.loss_metric.result().numpy())

    def save(self, save_path):
        tf.saved_model.save(self.neural_net, export_dir=save_path)


class LoadAgent:
    def __init__(self, load_path):
        # Load the neural network
        self.neural_net = tf.saved_model.load(load_path)

    def __call__(self, states):
        p, v = self.neural_net(states)
        return p.numpy()[0], v.numpy()[0][0]

