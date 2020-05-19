import gym
from game.RL_brain import DeepQNetwork
import numpy as np


def preprocess(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()


def run_game():
    step = 0
    for episode in range(300):
        observation = env.reset()
        # 预处理
        prev_x = None
        cur_x = preprocess(observation)
        observation = cur_x - prev_x if prev_x is not None else np.zeros_like(cur_x)
        prev_x = cur_x

        t = 0
        while True:
            env.render()

            action = RL.choose_action(observation)
            # 预处理
            observation_, reward, done, _ = env.step(action)
            cur_x = preprocess(observation_)
            observation_ = cur_x - prev_x if prev_x is not None else np.zeros_like(cur_x)
            prev_x = cur_x

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            observation = observation_
            t += 1
            if done:
                print("[*] Episode finished after {} timesteps".format(t+1))
                print("[*] reward: {}".format(reward))
                break
            step += 1


if __name__ == '__main__':
    env = gym.make("Pong-v0")
    n_actions = env.action_space.n
    n_features = 80*80
    RL = DeepQNetwork(n_actions,
                      n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000
                      )
    run_game()
    RL.plot_cost()

