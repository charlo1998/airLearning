
import gym
import tensorflow as tf
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm
from stable_baselines import A2C
import numpy as np

def tiny_filter_deep_nature_cnn(scaled_images, **kwargs):
    """
    CNN from Nature paper.
    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=6, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=8, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=10, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_4 = activ(conv(layer_3, 'c4', n_filters=12, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_5 = activ(conv(layer_4, 'c5', n_filters=14, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_5 = conv_to_fc(layer_5)
    layer_6 = activ(linear(layer_5, 'fc1', n_hidden=256, init_scale=np.sqrt(2)))
    layer_7 = activ(linear(layer_6, 'fc2', n_hidden=128, init_scale=np.sqrt(2)))
    return activ(linear(layer_7, 'fc3', n_hidden=128, init_scale=np.sqrt(2)))

# Custom MLP policy of two layers of size 128 each
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[64, 64],
                                                          vf=[64, 64, 64])],
                                           feature_extraction="mlp")


class CustomLSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=32, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[64, 'lstm', dict(vf=[32, 32], pi=[32])],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)

class CustomTinyDeepCNNPolicy(FeedForwardPolicy):
    
    def __init__(self, *args, **kwargs):
        super(CustomTinyDeepCNNPolicy, self).__init__(*args, **kwargs, cnn_extractor=tiny_filter_deep_nature_cnn, feature_extraction="cnn")

#dict(vf=[128, 64, 'lstm', 128, 128, 128], pi=[64, 128, 64])
## Create and wrap the environment
#env = gym.make('LunarLander-v2')
#env = DummyVecEnv([lambda: env])

#model = A2C(CustomPolicy, env, verbose=1)
## Train the agent
#model.learn(total_timesteps=100000)
## Save the agent
#model.save("a2c-lunar")

#del model
## When loading a model with a custom policy
## you MUST pass explicitly the policy when loading the saved model
#model = A2C.load("a2c-lunar", policy=CustomPolicy)