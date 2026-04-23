from src.env.wrapper import ManiSkillAgent
import numpy as np

def run_task():
    # 1. 实例化统一接口
    # TODO: 后续更换为你们自定义的 GraspCup-v0
    agent = ManiSkillAgent(env_id="PickCube-v1") 
    obs = agent.reset()
    
    print("🚀 开始执行主动感知任务...")
    for i in range(300):
        # 2. 动作策略层 (当前为正弦波探索，后续替换为郭浩轩的 DP 模型输出)
        action = np.sin(i / 10) * np.ones(agent.env.action_space.shape)
        
        # 3. 多模态感知层 (监控视觉不确定性)
        uncertainty = agent.get_visual_uncertainty(obs)
        if uncertainty > 0.5:
            # TODO: 这里后续接入刘蔚菡的触觉过滤逻辑
            print(f"Step {i}: ⚠️ 检测到视觉伪模糊 (方差: {uncertainty:.4f})，请求触觉修正！")
        
        # 4. 执行与环境交互
        obs, reward, done = agent.step(action)
        if done: break

    # 5. 渲染与结束
    agent.save_video("presentation_demo.mp4")
    agent.close()

if __name__ == "__main__":
    run_task()