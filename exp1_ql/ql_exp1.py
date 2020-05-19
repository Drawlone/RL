import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATES = 6
ACTIONS = ['left', 'right']
EPSILON = 0.9
ALPHA = 0.1
GAMA = 0.9
MAX_EPISODES = 13
FRESH_TIME = 0.3


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions,
    )
    print(table)
    return table


def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    return action_name


def get_env_feedback(S, A):
    if A == 'right':
        if S == N_STATES -2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R  # return new state and reward


def update_env(S, episode, step_counter):
    env_list = ['-']*(N_STATES-1) + ['T']
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                        ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 2
        is_terminal = False
        update_env(S, episode, step_counter)
        while not is_terminal:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + GAMA * q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminal = True

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            S = S_

            update_env(S_, episode, step_counter+1)
            step_counter += 1
    return q_table
# iloc按表格序号索引，loc按表格标签索引

# 1.在状态s'时，只是计算了 在s'时要采取哪个a'可以得到更大的Q值，并没有真的采取这个动作a'。
# 2.动作a的选取是根据当前Q网络以及策略（e-greedy），即每一步都会根据当前的状况选择一个动作A，目标Q值的计算是根据Q值最大的动作a'计算得来，因此为off-policy学习。


if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)

