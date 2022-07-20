import numpy as np
import pickle, os


class TwoAgentRecorder:
    def __init__(self, directory):
        self.directory = directory
        self.create_directory(directory)
        
        self.mean_total_rewards = []
        self.u_noise_thresholds = []
        self.v_noise_thresholds = []
        self.u_best_total_rewards = []
        self.v_best_total_rewards = []
        self.u_best_session = None
        self.v_best_session = None
        
        return None
    
    def create_directory(self, directory):
        pre_directory, _ = os.path.split(directory)
        if not os.path.exists(pre_directory):
            os.mkdir(pre_directory)
        if not os.path.exists(directory):
            os.mkdir(directory)
        return None
    
    def record_list(self, directory, data):
        with open(directory, 'wb') as file:
            pickle.dump(data, file)
        return None
    
    def record(self, env, u_agent, v_agent, episode, sessions):
        
        total_rewards = np.array([sum(session['rewards']) for session in sessions])
        mean_total_reward = np.mean(total_rewards)
        argmax_total_reward = np.argmax(total_rewards)
        argmin_total_reward = np.argmin(total_rewards)
        
        if self.u_best_session:
            if total_rewards[argmin_total_reward] < np.min(self.u_best_total_rewards):
                self.u_best_session = sessions[argmin_total_reward]  
                self.u_best_total_rewards.append(total_rewards[argmin_total_reward])
            else:
                self.u_best_total_rewards.append(np.min(self.u_best_total_rewards))
        else:
            self.u_best_session = sessions[argmin_total_reward]
            self.u_best_total_rewards.append(total_rewards[argmin_total_reward])

        if self.v_best_session:
            if total_rewards[argmax_total_reward] > np.max(self.v_best_total_rewards):
                self.v_best_session = sessions[argmax_total_reward]  
                self.v_best_total_rewards.append(total_rewards[argmax_total_reward])
            else:
                self.v_best_total_rewards.append(np.max(self.v_best_total_rewards))
        else:
            self.v_best_session = sessions[argmax_total_reward]
            self.v_best_total_rewards.append(total_rewards[argmax_total_reward])
        
        self.mean_total_rewards.append(mean_total_reward)
        self.u_noise_thresholds.append(u_agent.noise.threshold)
        self.v_noise_thresholds.append(v_agent.noise.threshold)

        np.save(self.directory + '/mean_total_rewards', self.mean_total_rewards)
        np.save(self.directory + '/u_best_total_rewards', self.u_best_total_rewards)
        np.save(self.directory + '/v_best_total_rewards', self.v_best_total_rewards)
        np.save(self.directory + '/u_noise_thresholds', self.u_noise_thresholds)
        np.save(self.directory + '/v_noise_thresholds', self.v_noise_thresholds)
        self.record_list(self.directory + '/u_best_session', self.u_best_session)
        self.record_list(self.directory + '/v_best_session', self.v_best_session)
        
        return None