from exp2_ql_sarsa.maze_env import Maze
from exp2_ql_sarsa.RL_brain import SarsaLambdaTable
from tqdm import tqdm

'''
def update_qlearing():
    for episode in range(100):
        observation = env.reset()
        while True:
            env.render()

            action = RL.choose_action(str(observation))

            observation_, reward, done = env.step(action)

            RL.learn(str(observation), action, reward, str(observation_))

            observation = observation_
            if done:
                break
    print('game over\n')
    env.destroy()
'''


def update_sarsa():
    n = 1
    for episode in tqdm(range(100)):
        observation = env.reset()
        epsilon = 1 - 1 / n
        print(epsilon)
        action = RL.choose_action(str(observation), epsilon)
        RL.eligibility_trace *= 0
        while True:
            env.render()
            observation_, reward, done = env.step(action)
            # print(reward)
            action_ = RL.choose_action(str(observation_), epsilon)
            RL.learn(str(observation), action, reward, str(observation_), action_)

            observation = observation_
            action = action_
            if done:
                break
        n += 1
    print('game over\n')
    env.destroy()
# 1.在状态s'时，就知道了要采取哪个a'，并真的采取了这个动作。
# 2.动作a的选取遵循e-greedy策略，目标Q值的计算也是根据（e-greedy）策略得到的动作a'计算得来，因此为on-policy学习。


if __name__ == "__main__":
    env = Maze()
    RL = SarsaLambdaTable(actions=list(range(env.n_actions)))
    env.after(100, update_sarsa)
    env.mainloop()