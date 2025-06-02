import gymnasium as gym
import highway_env
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import imageio

# 啟用 matplotlib 互動模式以即時顯示
plt.ion()

# 環境配置
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,  # 增加車輛數量
        "features": ["x", "y", "vx", "vy"],
        "normalize": True
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "duration": 100,
    "lanes_count": 4,  # 增加車道數
    "vehicles_density": 1.2,  # 增加車輛密度
    "controlled_vehicles": 1,
    "collision_reward": -50,
    "reward_speed_range": [25, 35],
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "high_speed_reward": 0.4,
    "lane_change_reward": -0.1,
    "right_lane_reward": 0.1,
    "safe_distance_reward": 0.2,  # 保持安全距離獎勵
    "acceleration_reward": 0.05,  # 平穩加速獎勵
    "deceleration_penalty": -0.05,  # 突然減速懲罰
    "overtaking_reward": 0.3,  # 成功超車獎勵
    "steady_speed_reward": 0.15,  # 保持穩定速度獎勵
    "off_road_penalty": -30,  # 偏離道路懲罰
    "time_reward": 0.1,  # 存活時間獎勵
    "road_width": 20,  # 增加道路寬度
    "vehicles_speed_range": [20, 40],  # 其他車輛的速度範圍
    "random_vehicle_speed": True,  # 允許其他車輛隨機速度
    "obstacle_probability": 0.3,  # 添加障礙物的概率
    "curved_road": True,  # 啟用彎道
    "road_curvature": 0.1  # 道路彎曲程度
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
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

# REINFORCE 訓練（含即時渲染）
def train(env, policy_net, optimizer, episodes=10, gamma=0.99):
    all_rewards = []
    # 創建 matplotlib 圖形用於即時顯示
    fig, ax = plt.subplots()
    img = ax.imshow(np.zeros((600, 600, 3), dtype=np.uint8))  # 調整為典型渲染尺寸
    plt.title("Highway Simulation")

    for episode in range(episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        total_reward = 0
        step = 0

        while True:
            # 每 2 步渲染一次以提高性能（可選）
            if step % 2 == 0:
                frame = env.render()
                img.set_data(frame)
                plt.pause(0.02)  # 調整為 0.02 以加快顯示

            # 將二維觀察扁平化
            state_tensor = torch.FloatTensor(state).flatten().unsqueeze(0)  # (5, 4) -> (20,) -> (1, 20)
            probs = policy_net(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            next_state, reward, terminated, truncated, _ = env.step(action.item())

            log_probs.append(log_prob)
            rewards.append(reward)
            total_reward += reward

            state = next_state
            step += 1
            if terminated or truncated:
                break

        # 計算折扣累積獎勵
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 損失計算與參數更新
        loss = -torch.stack(log_probs) * returns
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        all_rewards.append(total_reward)
        print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}")

    plt.ioff()  # 關閉互動模式
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
    # 修正 state_dim 計算
    state_dim = np.prod(env.observation_space.shape)  # 5 * 4 = 20
    action_dim = env.action_space.n

    policy_net = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

    # 訓練
    rewards = train(env, policy_net, optimizer, episodes=100)

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