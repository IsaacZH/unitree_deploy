## deploy_him_py 环境配置与运行说明

本文档补充 [deploy_real/inekf_odom_publisher_go2.py](deploy_real/inekf_odom_publisher_go2.py) 所需环境，重点包含以下 Python 依赖：

- unitree-sdk2py
- pinocchio
- inekf (来自仓库内 third_party/invariant-ekf 的 Python 绑定)

### 0. 克隆仓库时自动下载 third_party

推荐使用递归克隆（如果仓库配置了 submodule，会自动拉取 third_party）：

```bash
git clone --recurse-submodules <your-repo-url> /home/isaac/deploy_him_py
cd /home/isaac/deploy_him_py
git submodule update --init --recursive
```

可选：设置 git 全局递归子模块，后续 pull/clone 更省心：

```bash
git config --global submodule.recurse true
```

如果 third_party/invariant-ekf 没有被带下来，可手动补齐：

```bash
cd /home/isaac/deploy_him_py
mkdir -p third_party
if [ ! -d third_party/invariant-ekf/.git ]; then
	git clone https://github.com/inria-paris-robotics-lab/invariant-ekf.git third_party/invariant-ekf
fi
```

### 1. 创建并激活环境

建议使用 conda（与你当前 unitree_sdk2 工作流一致）：

```bash
conda create -n unitree_sdk2 python=3.11 -y
conda activate unitree_sdk2
```

### 2. 安装项目 Python 依赖

在仓库根目录执行：

```bash
cd /home/isaac/deploy_him_py
pip install -r requirements.txt
```

### 3. 安装 Pinocchio / EigenPy（INEKF Python 绑定前置）

推荐从 conda-forge 安装，版本匹配更稳定：

```bash
conda install -c conda-forge pinocchio==3.9.0 eigenpy cmake ninja pkg-config -y
```

### 4. 编译并安装 inekf Python 绑定

本仓库已带源码 [third_party/invariant-ekf](third_party/invariant-ekf)。

```bash
cd /home/isaac/deploy_him_py/third_party/invariant-ekf

cmake -S . -B build \
	-G Ninja \
	-DCMAKE_BUILD_TYPE=Release \
	-DBUILD_PYTHON_INTERFACE=ON \
	-DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
	-DPython3_EXECUTABLE="$(which python)"

cmake --build build -j
cmake --install build
```

说明：

- 使用 CMAKE_INSTALL_PREFIX=$CONDA_PREFIX 可避免写入系统目录。
- 安装后 inekf 模块会被放到当前 Python 环境 site-packages。

### 5. 依赖自检

```bash
python - <<'PY'
import pinocchio
import inekf
from inekf import InEKF, Kinematics, NoiseParams, RobotState
print('OK: pinocchio + inekf import success')
PY
```

### 6. 运行 INEKF Odom 发布器

文件位置：[deploy_real/inekf_odom_publisher_go2.py](deploy_real/inekf_odom_publisher_go2.py)

```bash
cd /home/isaac/deploy_him_py
python deploy_real/inekf_odom_publisher_go2.py wlp3s0 \
	--odom-topic rt/inekf/odom \
	--world-frame world \
	--base-frame-out base
```

如果没有安装 unitree_description 的 Python loader，会自动回退到 URDF 路径。可显式指定：

```bash
python deploy_real/inekf_odom_publisher_go2.py wlp3s0 \
	--urdf-path /home/isaac/deploy_him_py/go2_description/urdf/go2_description.urdf
```

### 7. 常用联调命令

键盘发布器：

```bash
python deploy_real/keyboard/keyboard_dds_publisher.py enp5s0 --topic rt/wireless_remote
```

主控制（Mujoco 链路）：

```bash
python deploy_real/deploy_real_go2.py enp5s0 go2.yaml --keyboard --mujoco
```

### 8. 常见问题

- 问题：No module named pinocchio
	- 处理：确认当前环境是 unitree_sdk2，并执行 conda install -c conda-forge pinocchio eigenpy

- 问题：No module named inekf
	- 处理：重新执行第 4 步，重点检查 CMAKE_INSTALL_PREFIX 与 Python3_EXECUTABLE 指向当前环境

- 问题：Failed to import unitree_description.loader.loadGo2
	- 处理：使用 --urdf-path 指定 [go2_description/urdf/go2_description.urdf](go2_description/urdf/go2_description.urdf)


