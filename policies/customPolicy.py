
import gym

from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C

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