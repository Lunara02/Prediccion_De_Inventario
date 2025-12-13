import numpy as np
import matplotlib.pyplot as plt
from export_canva import export_rewards_csv
from main import Tienda
import pickle

MAX_DEMANDA = 30

def run(episodes, is_training, render):

    env = Tienda(30)

    # Divide position and velocity into segments
    money_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 1000)    # Between -1.2 and 0.6
    stock_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 100)   
    price_space = np.linspace(env.observation_space.low[2], env.observation_space.high[2], 7)    # Between -0.07 and 0.07
    day_space = np.linspace(env.observation_space.low[3], env.observation_space.high[3], 31)    # Between -0.07 and 0.07

    if(is_training):
        q = np.zeros((len(money_space), len(stock_space), len(price_space), len(day_space), env.action_space.n))
    else:
        f = open('ql_q.pkl', 'rb')
        q = pickle.load(f)
        f.close()


    learning_rate_a = 0.01 # alpha or learning rate
    discount_factor_g = 0.99# gamma or discount factor.

    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05
    rng = np.random.default_rng()   # random number generator

    max_money = 0
    max_stock = 0

    rewards_per_episode = np.zeros(episodes)     # recompensa acumulada
    mean_rewards_per_episode = np.zeros(episodes) # promedio diario por episodio
    action_counts = np.zeros(env.action_space.n, dtype=int)


    for i in range(episodes):
        state = env.reset()[0]      # Starting position, starting velocity always 0
        state_m = np.digitize(state[0], money_space) - 1
        state_s = np.digitize(state[1], stock_space) - 1
        state_p = np.digitize(state[2], price_space) - 1
        state_d = np.digitize(state[3], day_space) - 1

        terminated = False          # True when reached goal

        total_rewards = 0
        daily_rewards = []

        while not terminated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_m, state_s, state_p, state_d, :])

            action_counts[action] += 1

            new_state,reward,terminated,_,_ = env.step(action)
            new_state_m = np.digitize(new_state[0], money_space) - 1
            new_state_s = np.digitize(new_state[1], stock_space) - 1
            new_state_p = np.digitize(state[2], price_space) - 1
            new_state_d = np.digitize(new_state[3], day_space) - 1

            
            if is_training:
                q[state_m, state_s, state_p, state_d, action] = q[state_m, state_s, state_p, state_d, action] + learning_rate_a * (
                reward + discount_factor_g*np.max(q[new_state_m, new_state_s, new_state_p, new_state_d, :]) - q[state_m, state_s, state_p, state_d, action]
                )


            state = new_state
            state_m = new_state_m
            state_s = new_state_s
            state_p = new_state_p
            state_d = new_state_d

            total_rewards += reward          # acumulado del episodio
            daily_rewards.append(reward) 

            if render:
                env.render()

            if max_money < money_space[state_m]:
                max_money = money_space[state_m]
            if max_stock < stock_space[state_s]:
                max_stock = stock_space[state_s]


        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        rewards_per_episode[i] = total_rewards
        mean_rewards_per_episode[i] = np.mean(daily_rewards)

        if render:
            print(f'FIN ITERACION {i + 1} QL')

    env.close()

    if is_training:
        f = open('ql_q.pkl','wb')
        pickle.dump(q, f)
        f.close()

    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])

    export_rewards_csv(rewards_per_episode, "ql_rewards_accumulated.csv")
    export_rewards_csv(mean_rewards_per_episode, "ql_rewards_mean.csv")
    export_rewards_csv(mean_rewards, "ql_mean_100.csv")
    

    plt.plot(mean_rewards_per_episode, label="Recompensa promedio por Meses")
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  
    plt.title("Q-LEARNING")                 # ← Título del gráfico
    plt.legend()
    plt.savefig("ql_rewards.png")

    print(max_money, max_stock)

    print(f"Max dinero alcanzado: {max_money}, Max stock alcanzado: {max_stock}, Max reward {np.max(rewards_per_episode)}")
    for a in range(env.action_space.n):
        print(f"Opción {a} : {action_counts[a]} veces")

    negative_sum = np.sum(rewards_per_episode[rewards_per_episode < 0])
    positive_sum = np.sum(rewards_per_episode[rewards_per_episode >= 0])

    negative_count = np.sum(rewards_per_episode < 0)
    positive_count = np.sum(rewards_per_episode >= 0)

    print(f"Cantidad de episodios con recompensas negativas: {negative_count}, acumulado: {negative_sum}")
    print(f"Cantidad de episodios con recompensas positivas: {positive_count}, acumulado: {positive_sum}")

if __name__ == '__main__':
    run(1000, is_training=False, render=False)

    # run(10000, is_training = True ,render=True)

    