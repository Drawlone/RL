import numpy as np
import pandas as pd


class RL(object):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        # self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype='float64')
        # self.n = 1
        # self.epsilon = 1/self.n

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, observation, epsilon):
        self.check_state_exist(observation)
        if np.random.uniform() < epsilon:  # eplsilon也可以先小后大，游戏开始随机探索，后面按q_table行动
            state_actions = self.q_table.loc[observation, :]
            state_actions = state_actions.reindex(np.random.permutation(state_actions.index))  # 防止value相等时，另一种action选不到
            action = state_actions.idxmax()
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass


class QLearningTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super().__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r

        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)


class SarsaTable(RL):
    def __init__(self, actions, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)


class SarsaLambdaTable(RL):
    def __init__(self, actions, learning_rate=0.1, reward_decay=0.9, e_greedy=0.6, trace_decay=0.9):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            to_be_append = pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            self.q_table = self.q_table.append(to_be_append)
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r
        error = q_target - q_predict
        # Method1
        # self.eligibility_trace.loc[s, :] += 1

        # Method2
        # self.eligibility_trace.loc[s, :] *= 0
        # self.eligibility_trace.loc[s, a] = 1

        self.eligibility_trace.loc[s, a] += 1
        # q update
        self.q_table += self.lr * error * self.eligibility_trace
        self.eligibility_trace *= self.gamma * self.lambda_




