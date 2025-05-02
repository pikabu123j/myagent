import os
import random
import numpy as np
from visualization.network_graph import NetworkVisualizer
from visualization.training_plot import TrainingMonitor

class SDWANController:
    def __init__(self, env_config: str, train_config: str):
        from .environment import SDWANEnvironment
        from .agent import DQNAgent
        from .llm_agents import LLMAgents
        
        self.env = SDWANEnvironment(env_config)
        self.agent = DQNAgent(
            state_size=len(self.env.get_state()),
            action_size=10  # Example: 10 possible paths
        )
        self.llm = LLMAgents()
        self._load_config(train_config)
        
        # Create outputs directory
        os.makedirs("outputs", exist_ok=True)

    def _load_config(self, path: str):
        import yaml
        with open(path) as f:
            config = yaml.safe_load(f)
        self.episodes = config['episodes']
        self.batch_size = config['batch_size']

    def train(self):
        net_viz = NetworkVisualizer(self.env.topology)
        train_viz = TrainingMonitor()
        
        for episode in range(self.episodes):
            state = self.env.get_state()
            total_reward = 0
            done = False
            
            # Get traffic profile (simplified)
            traffic = {"flow_count": random.randint(50, 200)}
            app_type = self.llm.analyze_traffic(traffic).get('dominant_application', 'data')
            
            while not done:
                action = self.agent.act(state)
                reward, done, metrics = self.env.step(action, app_type)
                
                # Capture every 10th episode for visualization
                if episode % 10 == 0:
                    net_viz.add_frame(metrics['path'])
                
                total_reward += reward
                state = self.env.get_state()
            
            train_viz.update(episode, total_reward, self.agent.epsilon)
            print(f"Episode {episode+1}/{self.episodes} | Reward: {total_reward:.2f} | ε: {self.agent.epsilon:.2f}")
        
        # Save visualizations
        net_viz.save_animation()
        train_viz.save_animation()