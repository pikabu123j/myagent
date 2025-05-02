import networkx as nx
import numpy as np
import yaml
from typing import Dict, List, Tuple

class SDWANEnvironment:
    def __init__(self, config_path: str):
        self.topology = nx.Graph()
        self._load_config(config_path)
        self.app_profiles = {
            'voip': {'max_latency': 50, 'min_throughput': 0.5, 'max_loss': 0.01},
            'video': {'max_latency': 100, 'min_throughput': 2.0, 'max_loss': 0.05},
            'data': {'max_latency': 300, 'min_throughput': 1.0, 'max_loss': 0.1}
        }

    def _load_config(self, path: str):
        with open(path) as f:
            config = yaml.safe_load(f)
        
        for node in config['nodes']:
            self.topology.add_node(node['id'], **node)
        
        for link in config['links']:
            self.topology.add_edge(
                link['source'], link['target'],
                latency=link['latency'],
                bandwidth=link['bandwidth'],
                loss=link['loss'],
                jitter=link['jitter']
            )

    def get_state(self) -> np.ndarray:
        return np.array([
            [data['latency'], data['bandwidth'], data['loss'], data['jitter']]
            for _, _, data in self.topology.edges(data=True)
        ]).flatten()

    def step(self, action: int, app_type: str) -> Tuple[float, bool, dict]:
        path = self._get_path(action)
        metrics = self._measure_path(path)
        reward = self._calculate_reward(metrics, self.app_profiles[app_type])
        done = self._check_qos_violation(metrics, self.app_profiles[app_type])
        return reward, done, metrics

    def _get_path(self, path_id: int) -> List[str]:
        all_paths = list(nx.all_simple_paths(
            self.topology, source='node1', target='node3'
        ))
        return all_paths[path_id % len(all_paths)]


    def _measure_path(self, path: List[str]) -> Dict[str, float]:
        latency = sum(self.topology[u][v]['latency'] for u, v in zip(path[:-1], path[1:]))
        bandwidth = min(self.topology[u][v]['bandwidth'] for u, v in zip(path[:-1], path[1:]))
        loss = 1 - np.prod([1 - self.topology[u][v]['loss'] for u, v in zip(path[:-1], path[1:])])
        return {'latency': latency, 'bandwidth': bandwidth, 'loss': loss, 'path': path}

    def _calculate_reward(self, metrics: Dict, qos: Dict) -> float:
        latency_norm = min(metrics['latency'] / qos['max_latency'], 1.0)
        throughput_norm = min(metrics['bandwidth'] / qos['min_throughput'], 1.0)
        loss_norm = min(metrics['loss'] / qos['max_loss'], 1.0)
        return 0.4*(1 - latency_norm) + 0.3*throughput_norm + 0.2*(1 - loss_norm)

    def _check_qos_violation(self, metrics: Dict, qos: Dict) -> bool:
        return (metrics['latency'] > qos['max_latency'] or 
                metrics['bandwidth'] < qos['min_throughput'] or 
                metrics['loss'] > qos['max_loss'])



