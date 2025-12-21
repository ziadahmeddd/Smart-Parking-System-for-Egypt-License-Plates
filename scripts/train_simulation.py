"""
Proper TD3 (Twin Delayed Deep Deterministic Policy Gradient) Implementation
for Smart Parking Allocation Training

This implements the full TD3 algorithm with:
- Twin Critic networks for value estimation
- Target networks with soft updates
- Delayed policy updates
- Exploration noise
- Experience replay buffer
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
from collections import deque
from typing import Tuple, List
import config

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --- 1. NEURAL NETWORK ARCHITECTURES ---
class Actor(nn.Module):
    """Policy network that outputs actions given states."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return torch.tanh(self.l3(x))  # Output in [-1, 1]

class Critic(nn.Module):
    """Q-value network that estimates action values."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture (twin)
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns both Q1 and Q2 values."""
        sa = torch.cat([state, action], dim=1)
        
        # Q1
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        # Q2
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        return q1, q2
    
    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns only Q1 value (used for actor loss)."""
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        return self.l3(q1)

# --- 2. REPLAY BUFFER ---
class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: float, reward: float, 
            next_state: np.ndarray, done: bool) -> None:
        """Store a transition."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)).unsqueeze(1),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)

# --- 3. VIRTUAL PARKING ENVIRONMENT ---
class VirtualParkingLot:
    """Simulated parking lot environment."""
    def __init__(self):
        self.num_spots = 3
        self.buildings = {"A": 0.0, "B": 4.0}
        self.spot_locations = {0: 1.0, 1: 2.0, 2: 3.0}
        self.sensor_state = []
        self.dest_loc = 0.0

    def reset(self) -> np.ndarray:
        """Reset environment and return initial state."""
        # Random occupancy (ensure at least one free spot)
        self.sensor_state = [random.randint(0, 1) for _ in range(self.num_spots)]
        if sum(self.sensor_state) == 0:
            self.sensor_state[random.randint(0, 2)] = 1
            
        # Random destination
        dest_key = random.choice(["A", "B"])
        self.dest_loc = self.buildings[dest_key]
        
        return np.array(self.sensor_state + [self.dest_loc], dtype=np.float32)

    def step(self, action_value: float) -> Tuple[np.ndarray, float, bool]:
        """
        Execute action and return (next_state, reward, done).
        
        Args:
            action_value: Continuous action in [-1, 1]
            
        Returns:
            next_state: Same as current state (episodic task)
            reward: Reward value
            done: Always True (single-step episodes)
        """
        # Decode action to spot index
        if action_value < config.ACTION_THRESHOLD_LOW:
            chosen_spot = 0
        elif action_value > config.ACTION_THRESHOLD_HIGH:
            chosen_spot = 2
        else:
            chosen_spot = 1
        
        # Calculate reward
        if self.sensor_state[chosen_spot] == 0:
            # Chose occupied spot - large penalty
            reward = config.PENALTY_OCCUPIED_SPOT
        else:
            # Chose free spot - reward based on distance
            spot_loc = self.spot_locations[chosen_spot]
            distance = abs(spot_loc - self.dest_loc)
            reward = 10.0 - (distance * config.REWARD_DISTANCE_MULTIPLIER)
        
        # Episode ends after one action (parking decision is one-shot)
        done = True
        next_state = self.reset()  # New scenario for next episode
        
        return next_state, reward, done

# --- 4. TD3 AGENT ---
class TD3Agent:
    """Twin Delayed Deep Deterministic Policy Gradient Agent."""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 0.001,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2
    ):
        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim, hidden_dim)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        
        self.total_iterations = 0
        
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> float:
        """Select action with optional exploration noise."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).item()
        
        if add_noise:
            noise = np.random.normal(0, 0.1)
            action = np.clip(action + noise, -1.0, 1.0)
            
        return action
    
    def train(self, replay_buffer: ReplayBuffer, batch_size: int = 64):
        """Train the agent on a batch of experiences."""
        if len(replay_buffer) < batch_size:
            return None, None
        
        self.total_iterations += 1
        
        # Sample batch
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        # --- Update Critic ---
        with torch.no_grad():
            # Select next action with target policy and add noise
            next_action = self.actor_target(next_state)
            noise = torch.randn_like(next_action) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_action = torch.clamp(next_action + noise, -1, 1)
            
            # Compute target Q-values (minimum of twin Q-values)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # Get current Q-values
        current_q1, current_q2 = self.critic(state, action)
        
        # Critic loss (MSE)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # --- Delayed Policy Update ---
        actor_loss = None
        if self.total_iterations % self.policy_delay == 0:
            # Compute actor loss
            actor_loss = -self.critic.q1(state, self.actor(state)).mean()
            
            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            
            actor_loss = actor_loss.item()
        
        return critic_loss.item(), actor_loss
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network parameters."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# --- 5. TRAINING LOOP ---
def train_td3(
    episodes: int = 10000,
    batch_size: int = 64,
    warmup_episodes: int = 100,
    print_every: int = 500
):
    """
    Main training loop for TD3 agent.
    
    Args:
        episodes: Number of training episodes
        batch_size: Batch size for training
        warmup_episodes: Episodes of random exploration before training
        print_every: Print progress every N episodes
    """
    print("🚀 Starting TD3 Training for Smart Parking Allocation")
    print(f"Episodes: {episodes}, Batch Size: {batch_size}")
    print("-" * 60)
    
    start_time = time.time()
    
    # Initialize environment and agent
    env = VirtualParkingLot()
    agent = TD3Agent(
        state_dim=config.TD3_STATE_DIM,
        action_dim=config.TD3_ACTION_DIM,
        hidden_dim=config.TD3_HIDDEN_DIM,
        lr=config.TD3_LEARNING_RATE
    )
    replay_buffer = ReplayBuffer()
    
    # Training metrics
    episode_rewards = []
    critic_losses = []
    actor_losses = []
    
    for episode in range(episodes):
        state = env.reset()
        
        # Select action (random for warmup, then from policy)
        if episode < warmup_episodes:
            action = np.random.uniform(-1, 1)
        else:
            action = agent.select_action(state, add_noise=True)
        
        # Execute action
        next_state, reward, done = env.step(action)
        
        # Store transition
        replay_buffer.add(state, action, reward, next_state, done)
        
        # Train agent
        if episode >= warmup_episodes:
            critic_loss, actor_loss = agent.train(replay_buffer, batch_size)
            if critic_loss is not None:
                critic_losses.append(critic_loss)
            if actor_loss is not None:
                actor_losses.append(actor_loss)
        
        episode_rewards.append(reward)
        
        # Progress reporting
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(episode_rewards[-print_every:])
            print(f"Episode {episode + 1:5d} | "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"Buffer: {len(replay_buffer):5d}")
    
    duration = time.time() - start_time
    
    # Final statistics
    print("-" * 60)
    print(f"✅ Training Complete in {duration:.2f} seconds")
    
    # Calculate performance metrics
    final_avg_reward = np.mean(episode_rewards[-1000:])
    initial_avg_reward = np.mean(episode_rewards[:1000])
    improvement = final_avg_reward - initial_avg_reward
    
    print(f"\n📈 Training Progress:")
    print(f"   Initial Reward (first 1000): {initial_avg_reward:.2f}")
    print(f"   Final Reward (last 1000)   : {final_avg_reward:.2f}")
    print(f"   Improvement                : {improvement:.2f} ({(improvement/abs(initial_avg_reward))*100:.1f}%)")
    
    if critic_losses:
        print(f"\n🧠 Network Performance:")
        print(f"   Final Critic Loss: {np.mean(critic_losses[-100:]):.4f}")
    if actor_losses:
        print(f"   Final Actor Loss : {np.mean(actor_losses[-100:]):.4f}")
    
    # Save model
    torch.save(agent.actor.state_dict(), config.TD3_MODEL_PATH)
    print(f"\n💾 Model saved to: {config.TD3_MODEL_PATH}")
    
    # Test the trained model
    print("\n" + "=" * 60)
    print("🧪 Testing Trained Model on Sample Scenarios")
    print("=" * 60)
    test_results = test_agent(agent, env)
    
    return final_avg_reward, test_results

def test_agent(agent: TD3Agent, env: VirtualParkingLot, num_tests: int = 50):
    """
    Test the trained agent on sample scenarios and calculate accuracy.
    
    Args:
        agent: Trained TD3 agent
        env: Virtual parking environment
        num_tests: Number of test scenarios
        
    Returns:
        Dictionary with test metrics
    """
    test_rewards = []
    correct_decisions = 0  # Chose a free spot
    optimal_decisions = 0   # Chose the best available spot
    
    for i in range(num_tests):
        state = env.reset()
        sensors = state[:3].astype(int).tolist()
        dest = "A" if state[3] < 2.0 else "B"
        dest_loc = state[3]
        
        action = agent.select_action(state, add_noise=False)
        _, reward, _ = env.step(action)
        test_rewards.append(reward)
        
        # Decode action to spot
        if action < config.ACTION_THRESHOLD_LOW:
            chosen_spot = 0
        elif action > config.ACTION_THRESHOLD_HIGH:
            chosen_spot = 2
        else:
            chosen_spot = 1
        
        # Check if decision was correct (chose a free spot)
        if sensors[chosen_spot] == 1:
            correct_decisions += 1
            
            # Check if it was optimal (closest free spot)
            spot_loc = [1.0, 2.0, 3.0][chosen_spot]
            
            # Find actual optimal spot
            best_dist = float('inf')
            for idx, is_free in enumerate(sensors):
                if is_free == 1:
                    dist = abs([1.0, 2.0, 3.0][idx] - dest_loc)
                    if dist < best_dist:
                        best_dist = dist
            
            current_dist = abs(spot_loc - dest_loc)
            if current_dist == best_dist:
                optimal_decisions += 1
        
        # Print first 10 tests
        if i < 10:
            spot_id = chosen_spot + 1
            status = "✅" if sensors[chosen_spot] == 1 else "❌"
            print(f"Test {i+1:2d}: Sensors={sensors}, Dest={dest} -> "
                  f"Spot {spot_id} {status} (action={action:.2f}, reward={reward:.1f})")
    
    # Calculate accuracy percentages
    avg_reward = np.mean(test_rewards)
    correctness_rate = (correct_decisions / num_tests) * 100
    optimality_rate = (optimal_decisions / num_tests) * 100
    
    print(f"\n{'='*60}")
    print("📊 TD3 Agent Performance Summary:")
    print("-" * 60)
    print(f"   Average Reward      : {avg_reward:.2f}/10")
    print(f"   Correctness Rate    : {correctness_rate:.1f}%  (Chose free spots)")
    print(f"   Optimality Rate     : {optimality_rate:.1f}%  (Chose best spots)")
    print(f"   Tests Run           : {num_tests}")
    print("-" * 60)
    
    # Performance grade
    print("\n💡 Agent Performance Grade:")
    if correctness_rate >= 95 and optimality_rate >= 80:
        print("   ✅ EXCELLENT - Agent makes smart decisions!")
    elif correctness_rate >= 90 and optimality_rate >= 70:
        print("   ✅ GOOD - Agent performs well")
    elif correctness_rate >= 80:
        print("   ⚠️  FAIR - Consider training longer")
    else:
        print("   ❌ POOR - Retrain with more episodes")
    
    # Save metrics
    metrics_dict = {
        "model": "td3_parking_agent",
        "average_reward": f"{avg_reward:.2f}",
        "correctness_rate": f"{correctness_rate:.1f}%",
        "optimality_rate": f"{optimality_rate:.1f}%",
        "tests_run": num_tests
    }
    
    with open("td3_agent_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)
    
    print(f"\n📊 Metrics saved to: td3_agent_metrics.json")
    print("=" * 60)
    
    return metrics_dict

if __name__ == "__main__":
    train_td3(
        episodes=config.TRAINING_EPISODES,
        batch_size=64,
        warmup_episodes=100,
        print_every=500
    )