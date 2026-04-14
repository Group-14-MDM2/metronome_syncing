

#from asyncio.windows_events import NULL
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


class Pendulum_on_SlidingPlatform():
    def __init__(self,nPendulums,MassConfig,LConfig,DampingConfig ,g = 9.8):
        #Assume the pendulum have same masses/radius, Ill add more functions later
        self.nPendulums = nPendulums   #Int
        self.SliderM,self.M = MassConfig
        self.L = LConfig
        self.TotalM = self.SliderM + np.sum(self.M)
        self.g = g
        self.PlatformDampingCoef,self.PendulumDampingCoef = DampingConfig
        self.statedim = 2+2*self.nPendulums

        
    def equations(self, t, state,ControlForce = 0):
        #States contain Phase and location for each pendulum and slider
        # SliderX, SliderV, Phase0, PhaseVelocity0.... etc, solveivp only takes 1D array

        SliderX = state[0]
        SliderV = state[1]

        PhaseStates = []
        for i in range(self.nPendulums):
            idx = 2 + 2*i
            PhaseStates.append([state[idx], state[idx+1]])
    
        SliderA = 0
        # A = sum n / m - sum d 
        n = 0
        d = 0
        for i in range(self.nPendulums):
            Phase = PhaseStates[i][0]
            PhaseV = PhaseStates[i][1]
            m = self.M[i]
            L = self.L[i]
            n += m * ( self.g * np.sin(Phase) * np.cos(Phase) +  L * ( PhaseV**2 ) * np.sin(Phase) )
            d += m * (np.cos(Phase) ** 2 )

        n -= self.PlatformDampingCoef * SliderV - ControlForce

        SliderA = n / (np.sum(self.M) + self.SliderM - d )
           

        PhaseA = np.zeros(self.nPendulums)

        for i in  range(self.nPendulums):
            Phase = PhaseStates[i][0]
            PhaseV = PhaseStates[i][1]
            m = self.M[i]
            L = self.L[i]
            PhaseA[i] = -self.g/L *np.sin(Phase) - SliderA/L*np.cos(Phase) - self.PendulumDampingCoef /( m * L**2 ) * PhaseV



        dstate = np.zeros(self.statedim)
        dstate[0] = SliderV
        dstate[1] = SliderA

        for i in range(self.nPendulums):
            idx = 2 + 2*i
            dstate[idx] = PhaseStates[i][1]      
            dstate[idx+1] = PhaseA[i]            
    
        return dstate  

    def simulate(self, t_span, initial_conditions,controller = None):
        def ModelControlEquation(t, state):
            if controller != None:
                force = controller(state)
            else:
                force = 0

            return self.equations(t, state, ControlForce=force)

        sol = solve_ivp(ModelControlEquation, t_span, initial_conditions, 
                       method='RK45', dense_output=True)
        return sol

    def plot_trajectories(self, sol, figsize=(12, 8)):
  
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        t = sol.t
        slider_x = sol.y[0, :]
        slider_v = sol.y[1, :]
        

        axes[0].plot(t, slider_x, 'k-', label='Slider Position', linewidth=2)
        axes[0].set_ylabel('Slider Position')
        axes[0].legend()
        axes[0].grid(True)
        

        for i in range(self.nPendulums):
            idx = 2 + 2*i
            phase = sol.y[idx, :]
            phase_norm = (phase + np.pi) % (2 * np.pi) - np.pi
            axes[1].plot(t, phase_norm, label=f'Pendulum {i+1} Phase')
        
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Phase (rad)')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        return fig, axes
    def plot_phase_space(self, sol, figsize=(12, 4)):
        if self.nPendulums == 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            idx = 2
            phase = sol.y[idx, :]
            phase_v = sol.y[idx+1, :]
            ax.plot(phase, phase_v, 'b-', alpha=0.7)
            ax.set_xlabel('Phase')
            ax.set_ylabel('Angular Velocity')
            ax.set_title('Phase Space')
            ax.grid(True)
        else:
            fig, axes = plt.subplots(1, self.nPendulums, figsize=(4*self.nPendulums, 4))
            for i in range(self.nPendulums):
                idx = 2 + 2*i
                phase = sol.y[idx, :]
                phase_v = sol.y[idx+1, :]
                axes[i].plot(phase, phase_v, 'b-', alpha=0.7)
                axes[i].set_xlabel(f'Phase {i+1}')
                axes[i].set_ylabel(f'Angular Velocity {i+1}')
                axes[i].grid(True)
        
        plt.tight_layout()
        return fig
    def get_order(self,state):
        angles = state[2::2] 
        order = np.sum(np.exp(1j * angles)) / self.nPendulums
        return order         
    def get_order_total_coherence(self, state):
        angles = state[2::2]
        velocities = state[3::2]
    
        # Phase coherence using Kuramoto logic
        phase_order = np.abs(np.sum(np.exp(1j * angles)) / self.nPendulums)
    
        # Velocity coherence: normalize variance by expected maximum swing speed
        # Assuming max velocity is around 10.0 rad/s
        v_var = np.var(velocities)
        v_coherence = np.exp(-v_var / 5.0) # Smooth exponential penalty
    
        # Combine them: must have both high phase order AND same speeds
        total_order = phase_order * v_coherence
        return total_order
    def get_synchronize_time(self,sol,stable_therhold = 0.9):
        orders = np.array([self.get_order(sol.y[:, i], self.nPendulums) for i in range(sol.y.shape[1])])
        condition = orders >= stable_therhold
        if condition.any():
            return np.argmax(condition)
        else:
            return np.inf

# Run RL model with mechincal Pendulum_on_SlidingPlatform() as enviornment 


class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor: Outputs force
        self.actor = nn.Linear(hidden_dim, 1)
        
        # Critic::Expected state
        self.critic = nn.Linear(hidden_dim, 1)
    def forward(self, state):
        x = self.common(state)
        mu = torch.tanh(self.actor(x)) 
        value = self.critic(x)
        return mu, value

def preprocess_state(state, nPendulums):
    
    #Transforms raw state [SliderX, SliderV, Phase1, PhaseV1...] into normalized features using sin/cos for phases.
    
    slider_x = state[0] / 5.0  
    slider_v = state[1] / 5.0
    features = [slider_x, slider_v]
    
    for i in range(nPendulums):
        theta = state[2 + 2*i]
        theta_v = state[3 + 2*i] / 10.0
        features.append(np.sin(theta))
        features.append(np.cos(theta))
        features.append(theta_v)
        
    return torch.FloatTensor(features)

def run_episode(model,env:Pendulum_on_SlidingPlatform, t_max = 10,dt = 0.05,specific_state = False):
    states, actions, rewards, values, log_probs = [], [], [], [], []

    #Init state
    if type(specific_state) == bool: 
        state = np.zeros(env.statedim)
        state[2::2] = np.random.uniform(-np.pi/2, np.pi/2, env.nPendulums)
    else:
        state = specific_state
    t = 0

    force_mult = 20
    alpha = 2.0 #Encourage Sychronize
    beta = 0.1 #Discourage energy spending, force**2 might be huge so we deccrease the factor
    theta = 5.0 #Time bonus, if the model sychronize realy they get a huge reward
    
    sychronize_therhold = 0.9
    sychronize_T_therhold = 2.5
    sychronize_T = 0
    while t < t_max:
        #Get model outputs and normalize
        feat = preprocess_state(state, env.nPendulums)
        mu, val = model(feat)

        dist = torch.distributions.Normal(mu, 0.1) 
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        f_val = action.item() * force_mult

        sol = solve_ivp(
            fun=lambda _t, _y: env.equations(_t, _y, ControlForce=f_val),
            t_span=[t, t + dt], y0=state, method='RK45'
        )
        next_state = sol.y[:, -1]

        # Calculate specialized reward for stability
        order = np.abs(env.get_order(next_state))
        
        # Combined Reward
        # We use order_mag**2 or higher to make the gradient steeper near synchronization
        r = alpha * (order**2) - beta * (action.item()**2)

        # Store for update
        states.append(feat)
        actions.append(action)
        rewards.append(r)
        values.append(val)
        log_probs.append(log_prob)
        
        state = next_state
        t += dt
        if np.abs(order) >= sychronize_therhold: # If sychronize give a huge bonus reward
            sychronize_T += dt
        else:
            sychronize_T = 0

        if sychronize_T >= sychronize_T_therhold:
            bonus = (t_max - t) * theta
            rewards[-1] += bonus
            print(f"Triggered bonus at{t}")
            break
    return rewards, values, log_probs


def train_model(model, optimizer, episode_data, gamma=0.98):
    rewards, values, log_probs = episode_data
    # 1. Compute Discounted Returns (Targets for Critic)
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    
    returns = torch.FloatTensor(returns)
    values = torch.cat(values).squeeze()
    log_probs = torch.cat(log_probs).squeeze()
    
    # 2. Advantage Estimation: A(s,a) = G - V(s)
    # Advantage tells us if the action was better than average for that state
    advantages = returns - values.detach()
    
    # 3. Actor Loss: Policy Gradient with Advantage
    actor_loss = -(log_probs * advantages).mean()
    
    # 4. Critic Loss: Value estimation error (MSE)
    critic_loss = F.mse_loss(values, returns)
    
    # Total combined loss
    loss = actor_loss + 0.5 * critic_loss
    
    optimizer.zero_grad()
    loss.backward()
    
    # Clip gradients to ensure numerical stability in ODE environments
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    
    return loss.item()

def test_model(initstate,env:Pendulum_on_SlidingPlatform ,model, t_max = 10,dt = 0.05):
    history = {'t': [], 'state': [], 'order': [], 'force': []}
    model.eval()
    #States were inputed externally, so differently trained model can experiment on same system, reduce performance differences due ot luck
    t = 0
    state = initstate.copy()
    t = 0
    force_mult = 10.0

    while t < t_max:
        feat = preprocess_state(state, env.nPendulums)

        with torch.no_grad():
            mu, _ = model(feat)
        f_val = mu.item() * force_mult
        sol = solve_ivp(
            fun=lambda _t, _y: env.equations(_t, _y, ControlForce=f_val),
            t_span=[t, t + dt],y0=state,method='RK45',rtol=1e-5,atol=1e-8  
        )# solveivp for a small dt, to get the next stage and reaction of the applied control force
        
        next_state = sol.y[:, -1]
        order = env.get_order(next_state)
        
        history['t'].append(t)
        history['state'].append(state.copy())
        history['order'].append(order)
        history['force'].append(f_val)
        state = next_state
        t += dt
    return history
    
def show_test_model(history,env:Pendulum_on_SlidingPlatform):
    t = np.array(history['t'])
    orders = np.array(history['order'])
    forces = np.array(history['force'])
    states = np.array(history['state'])

    dt = history['t'][1] - history['t'][0]
    velocities = np.array([s[1] for s in history['state']])
    total_work = np.sum(np.abs(forces * velocities)) * dt
    print(f"Total Work: {total_work:.2f} J")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)# plot order/sychronization progress
    axes[0].plot(t, abs(orders), 'r-', linewidth=2)
    axes[0].axhline(y=0.9, color='k', linestyle='--', label='Threshold')
    axes[0].set_ylabel('Order Parameter |R|')
    axes[0].set_title('Synchronization Progress')
    axes[0].grid(True)
    for i in range(env.nPendulums): # plot phases of pendulums
        phases = states[:, 2 + 2*i]
        phases = (phases + np.pi) % (2 * np.pi) - np.pi
        axes[1].plot(t, phases, label=f'Pendulum {i+1}')
    axes[1].set_ylabel('Phase (rad)')
    axes[1].legend(loc='right')
    axes[1].grid(True)

    axes[2].plot(t, forces, 'g-')#Plot control force
    axes[2].set_ylabel('Control Force (N)')
    axes[2].set_xlabel('Time (s)')
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

def save_checkpoint(model, optimizer, episode, path="ac_model.pth"):
    """
    Save the model, optimizer state, and current episode number.
    """
    checkpoint = {
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, path)
    print(f"---> Model saved to {path} at episode {episode}")

if __name__ == "__main__":
    env = Pendulum_on_SlidingPlatform(nPendulums=5, MassConfig=[0.5, [0.1, 0.1, 0.1 ,0.1 ,0.1]],LConfig=[1,1,1,1,1],DampingConfig= [0.01, 0.01])
    input_dim = 2 + 3 * env.nPendulums
    model = ActorCritic(input_dim=input_dim, hidden_dim=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    # Fixed initial state for progress tracking
    test_state = np.zeros(env.statedim)
    test_state[2::2] = np.random.uniform(-np.pi, np.pi, env.nPendulums)
    best_reward = -float('inf')
    History = test_model(test_state,env,model)
    show_test_model(History,env)
    for ep in range(2000):
        # Training Phase (uses randomized initial states for generalization)
        data = run_episode(model, env)
        loss_val = train_model(model, optimizer, data)
        
        current_total_reward = sum(data[0])
        # Evaluation Phase (uses fixed state to see learning progress)


        # Optional: Save a regular checkpoint every 500 episodes
        if ep % 500 == 0:
            save_checkpoint(model, optimizer, ep, f"checkpoint_ep{ep}.pth")
            #History = test_model(test_state,env,model)
            #show_test_model(History,env)

        if ep % 10 == 0:
            print(f"Episode {ep} | Loss: {loss_val:.4f} | Reward: {current_total_reward:.2f}")
    History = test_model(test_state,env,model)
    show_test_model(History,env)



    