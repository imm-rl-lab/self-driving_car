import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output


class ArrivalCarVisualizer:
    def __init__(self, waiting_for_show=10):
        self.waiting_for_show = waiting_for_show
        self.total_rewards = []
        self.noise_thresholds = []
        self.actions = []
        
    def show_fig(self, env, agent, sessions):
        
#         states = np.array([np.mean([session['states'][j] for session in sessions], axis=0) 
#                            for j in range(len(sessions[0]['states']))])
        max_number_of_actions = max([len(session['actions']) for session in sessions])
        actions = np.array([np.nanmean(
            [session['actions'][j] if j<len(session['actions']) else 
             np.nan for session in sessions]) for j in range(max_number_of_actions)])    
        
        plt.figure(figsize=[18, 6])
        plt.subplot(231)
        plt.plot(actions,'g', label='actions')
        plt.legend()
        plt.grid()

        plt.subplot(232)
        mean_total_rewards = np.mean(self.total_rewards[-20:])
        label = f'total_rewards: \n current={self.total_rewards[-1]:.2f} \n mean={mean_total_rewards:.2f}'
        plt.plot(self.total_rewards, 'g', label=label)
        plt.legend()
        plt.grid()

        plt.subplot(233)
        plt.plot(self.noise_thresholds,'g', label='noise_thrasholds')
        plt.legend()
        plt.grid()

        plt.subplot(234)
        plt.plot(env.sigmaReLe_array, label='SigmaRL')
        plt.legend()
        plt.grid()


        plt.subplot(235)
        plt.plot(env.FxRL_array, label='FxRL')
        plt.legend()
        plt.grid()
        
        
        plt.subplot(236)
        plt.plot(env.FzRL_array, label='FzRL')
        plt.legend()
        plt.grid()
       
        clear_output(True)
        
        plt.show()
        
#         print(f'sigma = {env.sigma_Fx}. Fz = {env.Fz_sigma}. Fx = {env.Fx_sigma}.  state = {env.state_array}')

        
    def show(self, env, agent, episode, sessions):
        total_reward = np.mean([sum(session['rewards']) for session in sessions])
        self.total_rewards.append(total_reward)
        self.noise_thresholds.append(agent.noise.threshold)
        
        if episode % self.waiting_for_show ==0:
            self.show_fig(env, agent, sessions)
            
    def clean(self):
        self.total_rewards = []
        self.noise_thresholds = []