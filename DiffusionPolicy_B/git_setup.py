#!/usr/bin/env python3
"""一键初始化 Squint 项目"""

import os
import subprocess
import sys

def run_cmd(cmd, description=""):
    """执行命令并打印"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"$ {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ 失败: {description}")
        sys.exit(1)
    print(f"✅ 完成: {description}")

def main():
    work_dir = os.path.expanduser("~/workspace")
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)
    
    # ============================================
    # 步骤 1: 克隆 Squint
    # ============================================
    if not os.path.exists("squint"):
        run_cmd(
            "git clone https://github.com/aalmuzairee/squint.git",
            "克隆 Squint 项目"
        )
    else:
        print("✅ Squint 已存在，跳过克隆")
    
    # ============================================
    # 步骤 2: 克隆你的团队项目
    # ============================================
    if not os.path.exists("Fanal_project_of_Robotics"):
        run_cmd(
            "git clone https://github.com/Leo0902110/Fanal_project_of_Robotics.git",
            "克隆团队项目"
        )
    else:
        print("✅ 团队项目已存在，跳过克隆")
    
    # ============================================
    # 步骤 3: 创建虚拟环境
    # ============================================
    run_cmd(
    f'"{sys.executable}" -m venv .venv',
    "创建 Python 3.10 虚拟环境"
    )
    
    # ============================================
    # 步骤 4: 激活虚拟环境并升级 pip
    # ============================================
    venv_bin = os.path.abspath(os.path.join(".venv", "bin" if os.name != "nt" else "Scripts"))
    pip_cmd = os.path.join(venv_bin, "pip")
    python_exe = os.path.join(venv_bin, "python.exe" if os.name == "nt" else "python")

    run_cmd(
        f"\"{python_exe}\" -m pip install --upgrade pip",
        "升级 pip"
    )
    
    # ============================================
    # 步骤 5: 安装 PyTorch（CUDA 12.4）
    # ============================================
    # run_cmd(
    #     f"\"{python_exe}\" -m pip install torch==2.6.0 torchvision==0.21.0 "
    #     # f"--index-url https://download.pytorch.org/whl/cu124",

    #     f"-i https://pypi.tuna.tsinghua.edu.cn/simple "
    #     f"--trusted-host pypi.tuna.tsinghua.edu.cn",
    #     "安装 PyTorch 2.6.0 (CUDA 12.4)"
    # )
    
# ============================================
    # 步骤 5: 安装 PyTorch（CUDA 12.4）
    # ============================================
    # 核心修改：移除原有的 -i 参数，改用 --index-url 强制指向官方 CUDA 仓库。
    # 这是防止清华源默认安装 CPU 版本的唯一可靠方法。
    run_cmd(
        f"\"{python_exe}\" -m pip install torch==2.6.0 torchvision==0.21.0 "
        f"--index-url https://pypi.tuna.tsinghua.edu.cn/simple"
        f"--extra-index-url https://download.pytorch.org/whl/cu124 ",
        "安装 PyTorch 2.6.0 (CUDA 12.4)"
    )
    
    # ============================================
    # 步骤 6: 进入 squint 目录，安装依赖
    # ============================================
    os.chdir("squint")
    
    # 修复1: 移除了不存在的 environment.txt 调用，直接硬编码依赖安装。
    # 修复2: 增加了 --extra-index-url 官方源，确保清华源缺失的 nightly 包能被顺利下载。
    run_cmd(
        f"\"{python_exe}\" -m pip install mani_skill_nightly gymnasium numpy tyro tqdm wandb opencv-python tensordict torchrl "
        f"-i https://pypi.tuna.tsinghua.edu.cn/simple "
        f"--extra-index-url https://pypi.org/simple "
        f"--trusted-host pypi.tuna.tsinghua.edu.cn",
        "安装 Squint 依赖"
    )
    
    os.chdir("..")
    
    # ============================================
    # 完成
    # ============================================
    print(f"\n{'='*60}")
    print("✅ 初始化完成！")
    print(f"{'='*60}")
    print("\n接下来的步骤：")
    print("1. 激活虚拟环境：")
    print(f"   $ source .venv/bin/activate  # Linux/Mac")
    print(f"   $ Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process   # Windows")
    print(f"   $ .\\.venv\\Scripts\\activate    # Windows")
    print("\n2. 验证安装：")
    print(f"   $ cd squint")
    print(f"   $ python -c 'import mani_skill; print(mani_skill.__version__)'")
    print("\n3. 运行快速测试：")
    print(f"   $ python examples/visualize_sim.py")

if __name__ == "__main__":
    main()
