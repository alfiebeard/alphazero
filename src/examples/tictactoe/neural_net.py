import tensorflow as tf
from keras.activations import relu
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout
from keras.losses import CategoricalCrossentropy, MSE
from keras.optimizers import Adam
from keras.metrics import Mean
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay


class TicTacToeNeuralNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.state_size = (3, 3, 1)
        self.action_size = 9

        # Decaying learning rate
        self.alpha = ExponentialDecay(initial_learning_rate=1e-2, decay_steps=2000, decay_rate=0.5, staircase=True)

        self.dropout_rate = 0.2
        self.num_channels = 512

        self.L1_dense_nodes = 256
        self.L2_dense_nodes = 128

        self.L1_policy_nodes = 50
        self.L2_policy_nodes = 30
        self.L3_policy_nodes = 20

        self.L1_value_nodes = 50
        self.L2_value_nodes = 40

        # Convolution layers
        self.L1_conv = Conv2D(self.num_channels, self.state_size[0], input_shape=self.state_size, padding='same')
        self.L1_conv_batch_norm = BatchNormalization()
        self.L2_conv = Conv2D(self.num_channels, self.state_size[0], padding='same')
        self.L2_conv_batch_norm = BatchNormalization()
        self.L3_conv = Conv2D(self.num_channels, self.state_size[0], padding='same')
        self.L3_conv_batch_norm = BatchNormalization()
        self.L4_conv = Conv2D(self.num_channels, self.state_size[0], padding='valid')
        self.L4_conv_batch_norm = BatchNormalization()

        # Dense layers
        self.L1_flatten = Flatten()
        self.L1_dense = Dense(self.L1_dense_nodes)
        self.L1_dense_batch_norm = BatchNormalization()
        self.L2_dense = Dense(self.L2_dense_nodes)
        self.L2_dense_batch_norm = BatchNormalization()

        self.dropout = Dropout(self.dropout_rate)

        # Policy head
        self.L1_policy_dense = Dense(self.L1_policy_nodes, activation='relu')
        self.L2_policy_dense = Dense(self.L2_policy_nodes, activation='relu')
        self.L3_policy_dense = Dense(self.L3_policy_nodes, activation='relu')
        self.policy_out = Dense(self.action_size, activation='softmax')

        # Value head
        self.L1_value = Dense(self.L1_value_nodes, activation='relu')
        self.L2_value = Dense(self.L2_value_nodes, activation='relu')
        self.value_out = Dense(1, activation='tanh')

        # Loss functions
        self.policy_loss = CategoricalCrossentropy()
        self.value_loss = MSE

        self.optimizer = Adam(learning_rate=self.alpha)

        self.policy_loss_metric = Mean(name='total_policy_loss')
        self.value_loss_metric = Mean(name='total_value_loss')
        self.loss_metric = Mean(name='total_loss')

    def core_network(self, x, training):
        x = self.L1_conv(x)
        x = self.L1_conv_batch_norm(x, training=training)
        x = relu(x)

        x = self.L2_conv(x)
        x = self.L2_conv_batch_norm(x, training=training)
        x = relu(x)

        x = self.L3_conv(x)
        x = self.L3_conv_batch_norm(x, training=training)
        x = relu(x)

        x = self.L4_conv(x)
        x = self.L4_conv_batch_norm(x, training=training)
        x = relu(x)

        x = self.L1_flatten(x)

        x = self.L1_dense(x)
        x = self.L1_dense_batch_norm(x, training=training)
        x = relu(x)
        x = self.dropout(x, training=training)

        x = self.L2_dense(x)
        x = self.L2_dense_batch_norm(x, training=training)
        x = relu(x)
        return self.dropout(x, training=training)

    def policy_head(self, x, training):
        x_policy = self.L1_policy_dense(x)
        x_policy = self.dropout(x_policy, training=training)
        x_policy = self.L2_policy_dense(x_policy)
        x_policy = self.dropout(x_policy, training=training)
        x_policy = self.L3_policy_dense(x_policy)
        x_policy = self.dropout(x_policy, training=training)
        return self.policy_out(x_policy)

    def value_head(self, x, training):
        x_value = self.L1_value(x)
        x_value = self.dropout(x_value, training=training)
        x_value = self.L2_value(x_value)
        x_value = self.dropout(x_value, training=training)
        return self.value_out(x_value)

    def call(self, x, training=False):
        x = self.core_network(x, training=training)
        p_head = self.policy_head(x, training=training)
        v_head = self.value_head(x, training=training)
        return p_head, v_head

    def loss(self, state, target_policy, target_value, training=False):
        with tf.GradientTape() as tape:
            policy, value = self.call(state, training=training)
            policy_loss = self.policy_loss(target_policy, policy)
            value_loss = self.value_loss(target_value, value)
            loss = tf.add(policy_loss, value_loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss, policy_loss, value_loss

