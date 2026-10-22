import torch
from go1_gym_learn.ppo_cse_automatic.arm_ac import ArmActorCritic
import os.path as osp
import pickle as pkl
from go1_gym.envs.automatic.legged_robot_config import Cfg
from go1_gym.envs.automatic import HistoryWrapper

def load_dog_policy(logdir, ckpt_id, Cfg, ckpt_file=None):
    from go1_gym_learn.ppo_cse_automatic.dog_ac import DogActorCritic
    actor_critic = DogActorCritic(
        Cfg.dog.dog_num_observations,
        Cfg.dog.dog_num_privileged_obs,
        Cfg.dog.dog_num_obs_history,
        Cfg.dog.dog_actions
    ).to("cpu")

    device = torch.device("cpu")
    if ckpt_file is not None:
        ckpt_path = ckpt_file
    else:
        if ckpt_id == 'last':
            ckpt_id_ = ckpt_id + '_dog'
        else:
            ckpt_id_ = ckpt_id.zfill(6)
        ckpt_path = logdir + f'/checkpoints_dog/ac_weights_{str(ckpt_id_)}.pt'

    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"[Load Policy] Dog checkpoint: {ckpt_path}")
    actor_critic.load_state_dict(ckpt)
    actor_critic.eval()

    def policy(obs, info={}):
        adaptation_module = actor_critic.adaptation_module
        body = actor_critic.actor_body
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_arm_policy(logdir, ckpt_id, Cfg, ckpt_file=None):
    actor_critic = ArmActorCritic(
        Cfg.arm.arm_num_observations,
        Cfg.arm.arm_num_privileged_obs,
        Cfg.arm.arm_num_obs_history,
        Cfg.arm.num_actions_arm_cd,
        device='cpu'
    ).to('cpu')

    device = torch.device("cpu")
    if ckpt_file is not None:
        ckpt_path = ckpt_file
    else:
        if ckpt_id == 'last':
            ckpt_id_ = ckpt_id + '_arm'
        else:
            ckpt_id_ = ckpt_id.zfill(6)
        ckpt_path = logdir + f'/checkpoints_arm/ac_weights_{str(ckpt_id_)}.pt'
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"[Load Policy] Arm checkpoint: {ckpt_path}")
    actor_critic.load_state_dict(ckpt)
    actor_critic.eval()
    adaptation_module = actor_critic.adaptation_module
    body = actor_critic.actor_body
    actor_his = actor_critic.actor_history_encoder

    def policy(obs, info={}):
        hist = actor_his.forward(obs["obs_history"].to('cpu')[..., :-Cfg.arm.arm_num_observations])
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs"].to('cpu'), latent, hist), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_unified_policy(logdir, ckpt_id, Cfg, ckpt_file=None):
    from go1_gym_learn.ppo_cse_unified.unified2head_ac import Unified2ActorCritic

    actor_critic = Unified2ActorCritic(
        Cfg.env.num_observations,
        Cfg.env.num_privileged_obs,
        Cfg.env.num_obs_history,
        Cfg.env.num_actions,
    ).to("cpu")

    device = torch.device("cpu")
    if ckpt_file is not None:
        ckpt_path = ckpt_file
    else:
        if ckpt_id == "last":
            ckpt_path = osp.join(logdir, "checkpoints_unified", "ac_weights_last.pt")
        else:
            ckpt_path = osp.join(logdir, "checkpoints_unified", f"ac_weights_{ckpt_id.zfill(6)}.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"[Load Policy] Unified checkpoint: {ckpt_path}")
    actor_critic.load_state_dict(ckpt, strict=False)
    actor_critic.eval()

    def policy(obs, info={}):
        obs_cur = obs["obs"].to("cpu")
        obs_history = obs["obs_history"].to("cpu")
        latent = actor_critic.adaptation_module(obs_history)
        actor_input = torch.cat((obs_cur, latent), dim=-1)
        hidden = actor_critic.actor_body(actor_input)
        mean_dog = actor_critic.action_dog_head(hidden)
        mean_arm = actor_critic.action_arm_head(hidden)
        info["latent"] = latent
        return torch.cat([mean_dog, mean_arm], dim=-1)

    return policy


def load_env(logdir, wrapper, headless=False, device='cuda:0', num_envs=None):
    print('*'*10, logdir)

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        cfg = pkl_cfg["Cfg"]
        # print(pkl_cfg.keys())
        # print(cfg.keys())

        for key, value in cfg.items():
            # 只处理在 Cfg 中已存在的字段，避免意外键
            if not hasattr(Cfg, key):
                continue

            # 这几个是嵌套结构（dog/arm/hybrid），期望是 dict
            if key in ["dog", "arm", "hybrid"]:
                if not isinstance(cfg[key], dict):
                    # 训练时不太可能出现，但为了鲁棒性，直接覆盖
                    setattr(Cfg, key, cfg[key])
                    continue

                for key2, value2 in cfg[key].items():
                    if not isinstance(value2, dict):
                        setattr(getattr(Cfg, key), key2, value2)
                    else:
                        for key3, value3 in value2.items():
                            setattr(getattr(getattr(Cfg, key), key2), key3, value3)
            else:
                # 其它字段如果是标量（如 use_rot6d 等），直接赋值；
                # 如果是 dict（如 env/commands/reward_scales 等），展开到子字段
                if not isinstance(value, dict):
                    setattr(Cfg, key, value)
                else:
                    # Handle nested dict structures like reward_scales.
                    # For ParamsProto/PrefixProto objects, we need to set all attributes
                    target_obj = getattr(Cfg, key)
                    for key2, value2 in value.items():
                        # Try to set the attribute, even if it doesn't exist in the current definition
                        # This is important for backward/forward compatibility
                        try:
                            setattr(target_obj, key2, value2)
                        except AttributeError:
                            # If setting fails (e.g., read-only attribute), skip it
                            pass

    # 设置默认值，避免访问不存在的属性
    if not hasattr(Cfg, 'use_rot6d'):
        Cfg.use_rot6d = False
    
    # 如果 end_effector_name 没有被设置（为空字符串），根据 asset.file 自动设置
    if not hasattr(Cfg.asset, 'end_effector_name') or not Cfg.asset.end_effector_name:
        asset_file = Cfg.asset.file
        if 'arx5p2Go1' in asset_file:
            Cfg.asset.end_effector_name = "zarx_body6"
        elif 'go2_airbot' in asset_file:
            Cfg.asset.end_effector_name = "arm_link06"
        elif 'go2_piper' in asset_file:
            Cfg.asset.end_effector_name = "end_link"
        # 如果没有匹配的机器人类型，保持为空字符串（使用默认 ee_idx = 23）
    
    Cfg.terrain.mesh_type = "plane"
    if Cfg.terrain.mesh_type == "plane":
      Cfg.terrain.teleport_robots = False

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.domain_rand.randomize_end_effector_force = False

    Cfg.env.num_recording_envs = 1 if num_envs is None else int(num_envs)
    Cfg.env.num_envs = 1 if num_envs is None else int(num_envs)
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = False
    Cfg.asset.render_sphere = True
    Cfg.env.episode_length_s = 10000
    Cfg.commands.resampling_time = 10000
    # Cfg.domain_rand.lag_timesteps = 6
    # Cfg.domain_rand.randomize_lag_timesteps = True
    # Cfg.control.control_type = "actuator_net"
    Cfg.rewards.use_terminal_body_height = False
    Cfg.rewards.use_terminal_roll = False
    Cfg.rewards.use_terminal_pitch = False
    Cfg.hybrid.rewards.use_terminal_body_height = False
    Cfg.hybrid.rewards.use_terminal_roll = False
    Cfg.hybrid.rewards.use_terminal_pitch = False
    Cfg.arm.commands.T_traj = [20000, 30000]
    # Cfg.sim.physx["num_position_iterations"] = 8
    # Cfg.sim.physx["num_velocity_iterations"] = 8
    
    env = wrapper(sim_device=device, headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)
    # load policy



    return env, Cfg