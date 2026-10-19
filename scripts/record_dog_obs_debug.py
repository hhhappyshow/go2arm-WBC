#!/usr/bin/env python3
"""
按方案 A：加载与 play 相同的环境，在仿真循环中记录 dog 单帧 obs、obs_history（及元数据），
并生成每一观测维的语义标签（含 dof_names[k] 与腿关节相关的 q / qd / prev_action 块）。

说明：
- env.cfg.env.num_observation_history：全臂观测 HistoryWrapper.obs_history 用的历史步数（默认 15）。
- env.cfg.dog.dog_num_observation_history：dog 策略用的历史步数（默认 30）；dog 的 obs_history 长度
  = dog_num_observations * dog_num_observation_history。

用法示例：
  python scripts/record_dog_obs_debug.py --logdir runs/your_run/auto_train/xxx --num_steps 50 --out /tmp/dog_obs_dbg.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

# 与 play_by_key 一致，保证可导入
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from go1_gym.envs.automatic import KeyboardWrapper  # noqa: E402
from go1_gym.utils.global_switch import global_switch  # noqa: E402
from scripts.load_policy import load_env  # noqa: E402


def build_dog_obs_dim_labels(base_env) -> list[str]:
    """
    与 VelocityTrackingEasyEnv.get_dog_observations()（go1_gym/envs/automatic/__init__.py）拼接顺序一致，
    生成长度 = dog_num_observations 的字符串列表，便于对照单帧 obs[:, d]。

    腿关节相关块使用 base_env.dof_names[k]（k = 0 .. num_actions_loco-1），与 dof_pos/dof_vel/actions 下标一致。
    """
    cfg = base_env.cfg
    dn = base_env.dof_names
    nloc = int(base_env.num_actions_loco)
    leg = [str(dn[i]) for i in range(nloc)]

    labels: list[str] = []

    labels += [
        "projected_gravity[0]",
        "projected_gravity[1]",
        "projected_gravity[2]",
    ]
    for k in range(nloc):
        labels.append(f"{leg[k]} | (q - q_default) * obs_scales.dof_pos  [dof index k={k}]")
    for k in range(nloc):
        labels.append(f"{leg[k]} | dof_vel * obs_scales.dof_vel  [dof index k={k}]")
    for k in range(nloc):
        labels.append(f"{leg[k]} | previous policy action (loco)  [dof index k={k}]")

    if cfg.hybrid.use_vision:
        labels += [
            "commands_dog[0]*scale  (vx cmd)",
            "commands_dog[1]*scale  (vy cmd)",
            "commands_dog[2]*scale  (yaw rate cmd)",
            "commands_dog[3]*scale  (body pitch cmd)",
            "commands_dog[4]*scale  (body roll cmd)",
        ]
        # 具体视觉块长度依环境张量，此处与 get_dog_observations 一致用 obj 块大小
        n_pose = int(base_env.obj_obs_pose_in_ee.shape[-1])
        n_abg = int(base_env.obj_obs_abg_in_ee.shape[-1])
        labels += [f"obj_obs_pose_in_ee[{i}]" for i in range(n_pose)]
        labels += [f"obj_obs_abg_in_ee[{i}]" for i in range(n_abg)]
        labels += ["base_roll (rpy)", "base_pitch (rpy)"]
    else:
        idx_arm = 9 if cfg.use_rot6d else 6
        labels += [
            "commands_dog[0]*scale  (vx cmd)",
            "commands_dog[1]*scale  (vy cmd)",
            "commands_dog[2]*scale  (yaw rate cmd)",
            "commands_dog[3]*scale  (body pitch cmd)",
            "commands_dog[4]*scale  (body roll cmd)",
        ]
        labels += [f"commands_arm_obs[{i}] (arm cmd in obs / zeros if switch off)" for i in range(idx_arm)]
        labels += ["base_roll (rpy)", "base_pitch (rpy)"]

    if cfg.env.observe_two_prev_actions:
        na = int(base_env.num_actions)
        labels += [f"last_actions[{i}] (full action dim)" for i in range(na)]

    if cfg.env.observe_timing_parameter:
        labels += ["gait_indices (timing)"]

    if cfg.env.observe_clock_inputs:
        labels += [f"clock_inputs[{i}]" for i in range(4)]

    if cfg.env.observe_vel:
        if cfg.commands.global_reference:
            labels = [
                "root_lin_vel (world) * lin_vel_scale [0]",
                "root_lin_vel (world) * lin_vel_scale [1]",
                "root_lin_vel (world) * lin_vel_scale [2]",
                "base_ang_vel * ang_vel_scale [0]",
                "base_ang_vel * ang_vel_scale [1]",
                "base_ang_vel * ang_vel_scale [2]",
            ] + labels
        else:
            labels = [
                "base_lin_vel * lin_vel_scale [0]",
                "base_lin_vel * lin_vel_scale [1]",
                "base_lin_vel * lin_vel_scale [2]",
                "base_ang_vel * ang_vel_scale [0]",
                "base_ang_vel * ang_vel_scale [1]",
                "base_ang_vel * ang_vel_scale [2]",
            ] + labels

    if cfg.env.observe_only_ang_vel:
        labels = [
            "base_ang_vel * ang_vel_scale [0] (only_ang)",
            "base_ang_vel * ang_vel_scale [1] (only_ang)",
            "base_ang_vel * ang_vel_scale [2] (only_ang)",
        ] + labels

    if cfg.env.observe_only_lin_vel:
        labels = [
            "base_lin_vel * lin_vel_scale [0] (only_lin)",
            "base_lin_vel * lin_vel_scale [1] (only_lin)",
            "base_lin_vel * lin_vel_scale [2] (only_lin)",
        ] + labels

    if cfg.env.observe_yaw:
        labels += ["heading (yaw observation)"]

    if cfg.env.observe_contact_states:
        nf = int(base_env.contact_forces[:, base_env.feet_indices, 2].shape[1])
        labels += [f"foot_contact[{i}]" for i in range(nf)]

    return labels


def compute_leg_joint_obs_index_ranges(base_env) -> dict:
    """
    返回腿关节在单帧 dog obs 中的全局维下标（与 get_dog_observations 一致）。
    仅 (q-q0)、qd、prev_action 三块与 dof_names[k] 一一对应；k 为 0..num_actions_loco-1。
    """
    cfg = base_env.cfg
    nloc = int(base_env.num_actions_loco)
    d = 0
    if cfg.env.observe_vel:
        d += 6
    if cfg.env.observe_only_ang_vel:
        d += 3
    if cfg.env.observe_only_lin_vel:
        d += 3

    g0 = d
    d += 3  # projected_gravity
    q_start = d
    d += nloc
    qd_start = d
    d += nloc
    act_start = d
    d += nloc

    return {
        "gravity_dims": [g0, g0 + 1, g0 + 2],
        "delta_q_dims": [q_start + k for k in range(nloc)],
        "dof_vel_dims": [qd_start + k for k in range(nloc)],
        "prev_action_dims": [act_start + k for k in range(nloc)],
        "note": "obs[0, dim]；第 k 个腿关节：delta_q_dims[k]、(qd)、prev_action_dims[k] 对应 dof_names[k]",
    }


def verify_labels_match_cfg(base_env, labels: list[str]) -> None:
    expected = int(base_env.cfg.dog.dog_num_observations)
    got = len(labels)
    if got != expected:
        raise RuntimeError(
            f"Label count {got} != cfg.dog.dog_num_observations ({expected}). "
            "训练配置与当前 get_dog_observations 逻辑可能不一致，请核对 __init__.py 与 parameters.pkl。"
        )


def record_once(
    env,
    cfg,
    out_npz: str,
    num_steps: int,
) -> None:
    base = env.env if hasattr(env, "env") else env

    labels = build_dog_obs_dim_labels(base)
    verify_labels_match_cfg(base, labels)

    idx_map = compute_leg_joint_obs_index_ranges(base)

    obs_hist_frames = int(cfg.dog.dog_num_observation_history)
    n_dog = int(cfg.dog.dog_num_observations)
    n_full_hist = int(cfg.env.num_observation_history)

    leg_names = [str(base.dof_names[i]) for i in range(int(base.num_actions_loco))]

    traj_obs = []
    traj_hist = []
    traj_cmd = []

    obs_pack = env.reset()
    for _t in range(num_steps):
        dog = env.get_dog_observations()
        traj_obs.append(dog["obs"].detach().cpu().numpy())
        traj_hist.append(dog["obs_history"].detach().cpu().numpy())
        traj_cmd.append(base.commands_dog.detach().cpu().numpy())

        actions_dog = torch.zeros(env.num_envs, int(base.num_actions_loco), device=env.device)
        actions_arm = torch.zeros(env.num_envs, int(base.num_actions_arm), device=env.device)
        env.step(actions_dog, actions_arm)

    payload = {
        "obs_single_frame_stack": np.stack(traj_obs, axis=0),
        "dog_obs_history_stack": np.stack(traj_hist, axis=0),
        "commands_dog_stack": np.stack(traj_cmd, axis=0),
        "obs_dim_labels": np.array(labels, dtype=object),
        "dof_names_leg": np.array(leg_names, dtype=object),
        "leg_joint_delta_q_dims": np.array(idx_map["delta_q_dims"], dtype=np.int32),
        "leg_joint_dof_vel_dims": np.array(idx_map["dof_vel_dims"], dtype=np.int32),
        "leg_joint_prev_action_dims": np.array(idx_map["prev_action_dims"], dtype=np.int32),
        "gravity_obs_dims": np.array(idx_map["gravity_dims"], dtype=np.int32),
        "dog_num_observations": n_dog,
        "dog_num_observation_history": obs_hist_frames,
        "dog_num_obs_history": int(cfg.dog.dog_num_obs_history),
        "env_num_observation_history": n_full_hist,
        "env_num_observations_full": int(cfg.env.num_observations),
        "observe_vel": bool(cfg.env.observe_vel),
        "use_rot6d": bool(getattr(cfg, "use_rot6d", False)),
        "hybrid_use_vision": bool(cfg.hybrid.use_vision),
    }
    np.savez_compressed(out_npz, **payload)
    print(f"[record_dog_obs_debug] Wrote {out_npz}")
    print(f"  dog_num_observations = {n_dog}, dog_num_observation_history = {obs_hist_frames}")
    print(f"  dog obs_history flat length = {cfg.dog.dog_num_obs_history} (= {n_dog} * {obs_hist_frames})")
    print(f"  env.num_observation_history (full obs stack) = {n_full_hist}")
    print("  Leg joint ↔ obs dim (dof_names[k]，与单帧 obs 下标一致):")
    for k, name in enumerate(leg_names):
        print(
            f"    k={k} {name}:  Δq→{idx_map['delta_q_dims'][k]}  qd→{idx_map['dof_vel_dims'][k]}  prev_a→{idx_map['prev_action_dims'][k]}"
        )


def main():
    parser = argparse.ArgumentParser(description="Record dog obs / obs_history for debugging (scheme A).")
    parser.add_argument("--logdir", type=str, required=True, help="Run directory containing parameters.pkl")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--num_steps", type=int, default=30)
    parser.add_argument(
        "--out",
        type=str,
        default="dog_obs_debug.npz",
        help="Output .npz path",
    )
    args = parser.parse_args()

    global_switch.open_switch()
    env, cfg = load_env(args.logdir, wrapper=KeyboardWrapper, headless=args.headless, device=args.sim_device)
    record_once(env, cfg, args.out, args.num_steps)


if __name__ == "__main__":
    main()
