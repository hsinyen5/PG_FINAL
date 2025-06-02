import gymnasium as gym
import highway_env
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import imageio
from torch.distributions import Categorical

# 啟用 matplotlib 互動模式以即時顯示
plt.ion()

# PPO 超參數
PPO_EPSILON = 0.2  # PPO 裁剪參數
PPO_EPOCHS = 10    # PPO 更新次數
BATCH_SIZE = 64    # 批次大小
GAMMA = 0.99       # 折扣因子
LAMBDA = 0.95      # GAE 參數
ENTROPY_COEF = 0.01  # 熵係數
VALUE_COEF = 0.5   # 價值損失係數

# 環境配置
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 30,  # 減少車輛數量，降低碰撞風險
        "features": ["x", "y", "vx", "vy"],
        "normalize": True
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "duration": 300,  # 增加持續時間到 200 步5
    "lanes_count": 4,
    "vehicles_density": 1.0,  # 降低車輛密度
    "controlled_vehicles": 1,
    "collision_reward": -100,  # 大幅增加碰撞懲罰
    "reward_speed_range": [20, 30],  # 降低理想速度範圍，提高安全性
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "high_speed_reward": 0.1,  # 降低高速獎勵
    "lane_change_reward": -0.3,  # 增加變換車道懲罰
    "right_lane_reward": 0.0,
    "safe_distance_reward": 0.8,  # 大幅增加安全距離獎勵
    "acceleration_reward": 0.05,  # 降低加速獎勵
    "deceleration_penalty": -0.1,  # 降低減速懲罰
    "overtaking_reward": 0.15,  # 降低超車獎勵
    "steady_speed_reward": 0.2,
    "off_road_penalty": -50,  # 增加偏離道路懲罰
    "time_reward": 0.05,  # 降低時間獎勵
    "road_width": 20,
    "vehicles_speed_range": [15, 25],  # 降低其他車輛速度範圍
    "random_vehicle_speed": True,
    "obstacle_probability": 0.2,  # 降低障礙物概率
    "curved_road": True,
    "road_curvature": 0.05,  # 降低道路彎曲程度
    # 新增獎勵參數
    "smooth_acceleration_reward": 0.1,
    "smooth_steering_reward": 0.1,
    "constant_speed_reward": 0.15,
    "sudden_maneuver_penalty": -0.2,
    "lane_keeping_reward": 0.2,
    "speed_variance_penalty": -0.15,
    # 新增安全相關獎勵
    "emergency_brake_penalty": -0.5,  # 緊急煞車懲罰
    "safe_lane_change_reward": 0.3,  # 安全變道獎勵
    "collision_warning_penalty": -0.4,  # 碰撞警告懲罰
    "minimum_safe_distance": 20,  # 最小安全距離（米）
    "safe_distance_threshold": 30,  # 安全距離閾值（米）
}

# 建立環境
try:
    env = gym.make("highway-v0", render_mode="rgb_array", config=config)
    print("Environment loaded successfully.")
except Exception as e:
    print(f"Failed to initialize environment: {e}")
    exit(1)

# 策略網路
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.actor(x)

# 價值網路
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        return self.critic(x)

def compute_gae(rewards, values, next_value, dones, gamma=GAMMA, lambda_=LAMBDA):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value_t = next_value
        else:
            next_value_t = values[t + 1]
        
        delta = rewards[t] + gamma * next_value_t * (1 - dones[t]) - values[t]
        gae = delta + gamma * lambda_ * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + torch.tensor(values, dtype=torch.float32)
    return advantages, returns

def ppo_update(policy_net, value_net, optimizer, states, actions, old_probs, advantages, returns, epsilon=PPO_EPSILON):
    for _ in range(PPO_EPOCHS):
        # 計算新的動作概率和價值
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_probs = torch.FloatTensor(old_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # 計算新的動作概率
        new_probs = policy_net(states)
        dist = Categorical(new_probs)
        new_log_probs = dist.log_prob(actions)
        
        # 計算比率
        ratio = torch.exp(new_log_probs - torch.log(old_probs))
        
        # PPO 裁剪目標
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # 價值損失
        value_pred = value_net(states).squeeze()
        value_loss = nn.MSELoss()(value_pred, returns)
        
        # 熵損失
        entropy = dist.entropy().mean()
        
        # 總損失
        loss = actor_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def train(env, policy_net, value_net, optimizer, episodes=100):
    all_rewards = []
    fig, ax = plt.subplots()
    img = ax.imshow(np.zeros((600, 600, 3), dtype=np.uint8))
    plt.title("Highway Simulation")

    for episode in range(episodes):
        state, _ = env.reset()
        episode_rewards = []
        episode_values = []
        episode_actions = []
        episode_probs = []
        episode_dones = []
        episode_states = []  # 新增：收集狀態
        total_reward = 0
        step = 0

        while True:
            if step % 2 == 0:
                frame = env.render()
                img.set_data(frame)
                plt.pause(0.02)

            state_tensor = torch.FloatTensor(state).flatten().unsqueeze(0)
            with torch.no_grad():
                probs = policy_net(state_tensor)
                value = value_net(state_tensor)
            
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            
            episode_rewards.append(reward)
            episode_values.append(value.item())
            episode_actions.append(action.item())
            episode_probs.append(log_prob.exp().item())
            episode_dones.append(terminated or truncated)
            episode_states.append(state)  # 新增：保存當前狀態
            
            total_reward += reward
            state = next_state
            step += 1
            
            if terminated or truncated:
                break

        # 計算 GAE 和優勢
        with torch.no_grad():
            next_value = value_net(torch.FloatTensor(next_state).flatten().unsqueeze(0)).item()
        
        advantages, returns = compute_gae(
            episode_rewards, 
            episode_values, 
            next_value, 
            episode_dones
        )
        
        # PPO 更新
        ppo_update(
            policy_net,
            value_net,
            optimizer,
            np.array([state.flatten() for state in episode_states]),  # 使用收集的狀態
            episode_actions,
            episode_probs,
            advantages,
            returns
        )

        all_rewards.append(total_reward)
        print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}")

    plt.ioff()
    return all_rewards

# 測試並保存模擬影片
def save_simulation_video(env, policy_net, filename="highway_simulation.mp4", max_steps=100):
    try:
        writer = imageio.get_writer(filename, fps=15)
    except Exception as e:
        print(f"Failed to initialize video writer: {e}")
        print("Please install imageio-ffmpeg: pip install imageio-ffmpeg")
        return

    state, _ = env.reset()
    for _ in range(max_steps):
        frame = env.render()
        writer.append_data(frame)
        state_tensor = torch.FloatTensor(state).flatten().unsqueeze(0)
        probs = policy_net(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        state, _, terminated, truncated, _ = env.step(action.item())
        if terminated or truncated:
            break
    writer.close()
    print(f"Simulation video saved as {filename}")

# 主程式
if __name__ == "__main__":
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n

    policy_net = PolicyNetwork(state_dim, action_dim)
    value_net = ValueNetwork(state_dim)
    optimizer = optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=3e-4)

    # 訓練
    rewards = train(env, policy_net, value_net, optimizer, episodes=100)

    # 保存模擬影片
    save_simulation_video(env, policy_net, filename="highway_simulation.mp4")

    # 繪製獎勵曲線
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward")
    plt.grid()
    plt.show()

    env.close()