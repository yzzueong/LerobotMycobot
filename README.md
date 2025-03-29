# Lerobot for Mycobot 320

Lerobot is a open source project for teleoperation.
But the Mycobot 320 robot arm is not supported yet. So I modify this repo to support the way Mycobot 320 could collect data.

Mycobot 320 has a function called "Drag Teaching", so I change the teleoperation code and the way they collect data.

I setup two cameras in the environment, one is on top of the robot arm, the other is on the side (I didn't mount a camera on the robot arm).
Here's some key modifications I made:
1. lerobot.common.robot_devices.control_utils.control_loop_mycobot
2. lerobot.common.robot_devices.robots.utils.make_robot_from_config
3. lerobot.common.robot_devices.robots.utils.make_robot_config
4. lerobot.common.robot_devices.robots.mycobot_manipulator.MycobotManipulator
5. lerobot.common.robot_devices.robots.configs.MycobotRobotConfig

To run this project, you also need to follow Lerobot project's README.md.

## Installation

Download our source code:
```bash
git clone https://github.com/yzzueong/LerobotMycobot.git
cd lerobot
```

Create a virtual environment with Python 3.10 and activate it, e.g. with [`miniconda`](https://docs.anaconda.com/free/miniconda/index.html):
```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

Install 🤗 LeRobot, Mycobot and Pi0:
```bash
pip install -e ".[mycobot, pi0]"
```

To use [Weights and Biases](https://docs.wandb.ai/quickstart) for experiment tracking, log in with
```bash
wandb login
```

Set Hugging Face env:
```bash
huggingface-cli login --token <your_huggingface_token> --add-to-git-credential
# If you are using Windows, you can use the following command to set the environment variable:
$HF_USER = huggingface-cli whoami | Select-String -Pattern "^\S+" | ForEach-Object { $_.Matches.Value }
# If you are using Ubuntu,
HF_USER=$(huggingface-cli whoami | head -n 1)
```

## Detect Camera and get your camera id
```bash
python .\lerobot\common\robot_devices\cameras\opencv.py --images-dir outputs/images_from_opencv_cameras
```
After you run this command, you need to check the index of your camera and write a right index in lerobot.common.robot_devices.robots.configs.MycobotRobotConfig 

## Record data
To record dataset, you need to run the following command:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=mycobot \
  --robot.ip=your.robot.ip.address \
  --robot.port=9000 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="<your instruction of your task>" \
  --control.repo_id=${HF_USER}/mycobot_test \
  --control.tags='[\"mycobot\",\"tutorial\"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=15 \
  --control.push_to_hub=false
```
If your want to add more data to the existing dataset, you need to add one more line `--control.resume=true`

## Train ACT or Diffusion Policy
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=your/path/to/dataset \
  --dataset.root=your/path/to/dataset \
  --policy.type=act \
  --output_dir=outputs/train/act_mycobot \
  --job_name=act_mycobot \
  --policy.device=cuda \
  --policy.use_amp=true \
  --wandb.enable=false \
  --save_freq=10000 \
  --steps=100000 

python lerobot/scripts/train.py \
  --dataset.repo_id=your/path/to/dataset \
  --dataset.root=your/path/to/dataset \
  --policy.type=diffusion \
  --output_dir=outputs/train/dp_mycobot \
  --job_name=dp_mycobot \
  --policy.device=cuda \
  --policy.use_amp=true \
  --wandb.enable=true \
  --save_freq=1000 \
  --steps=40000

```

## Convert JAX pi0 model to Pytorch
There are two steps to finish this procedure.
```bash
python lerobot/common/policies/pi0/conversion_scripts/convert_pi0_to_hf_lerobot.py \
    --checkpoint_dir path/to/your/openpi/model/params \
    --output_path path/to/save/pytorch/model
```
After this procedure, you still cannot load the model directly, because the model still miss some parameters.
We need to load the model we just saved and train it a little bit using Lerobot code.
```bash
python lerobot/scripts/train.py \
  --policy.path=path/to/save/pytorch/model \
  --dataset.repo_id=huggingface/dataset/id \
  --steps=5
```

## Test your models
Before you test your model, you need to check parameters in `path/to/your/checkpoint/pretrained_model/config.json`.
ACT, Diffusion Policy and Pi0 use the same command. They use config.json to distinguish which model need to be built.
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=mycobot \
  --robot.ip=your.robot.ip.address \
  --robot.port=9000 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="<your instruction of your task>" \
  --control.repo_id=${HF_USER}/eval_act_mycobot \
  --control.warmup_time_s=3 \
  --control.episode_time_s=300 \
  --control.reset_time_s=30 \
  --control.num_episodes=3 \
  --control.push_to_hub=false \
  --control.policy.path=path/to/your/checkpoint/pretrained_model 

```