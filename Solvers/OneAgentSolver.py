def get_session(env, agent, session_len, agent_learning):
    session = {}
    session['states'], session['actions'], session['rewards'], session['dones'], session['info'] = [], [], [], [], []
    
    state = env.reset()
    session['states'].append(state)

    if agent_learning:
        agent.noise.reset()
        
    done = False

    for i in range(session_len):
        action = agent.get_action(state)
        session['actions'].append(action)

        state, reward, done, info = env.step(action)
        session['states'].append(state)
        session['rewards'].append(reward)
        session['dones'].append(done)
        session['info'].append(info)
        
        if done:
            break
    
    return session


def go(env, agent, show, episode_n=100, session_n=1, session_len=10000, agent_learning=True):

    for episode in range(episode_n):
        sessions = [get_session(env, agent, session_len, agent_learning) for i in range(session_n)]
        print(f'\r{episode} episode', end='', flush=True)
        show(env, agent, episode, sessions)
        
        if agent_learning:
            agent.fit(sessions)
            agent.noise.reduce()

    print()

    return None