import math
import time# 修复：设置LD_LIBRARY_PATH以找到Python共享库（解决libpython3.8.so.1.0找不到的问题）
import os
import sys
import ctypes
import ctypes.util

# 如果使用conda环境，将conda环境的lib目录添加到LD_LIBRARY_PATH
# 并使用ctypes预加载Python共享库，确保Isaac Gym能找到它
if 'CONDA_PREFIX' in os.environ:
    conda_lib_path = os.path.join(os.environ['CONDA_PREFIX'], 'lib')
    if conda_lib_path not in os.environ.get('LD_LIBRARY_PATH', ''):
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = f"{conda_lib_path}:{current_ld_path}" if current_ld_path else conda_lib_path
    
    # 预加载Python共享库，确保Isaac Gym的C扩展模块能找到它
    # 这需要在导入isaacgym之前完成
    python_lib_path = os.path.join(conda_lib_path, 'libpython3.8.so.1.0')
    if os.path.exists(python_lib_path):
        try:
            ctypes.CDLL(python_lib_path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            # 如果加载失败，尝试其他可能的路径
            pass

import isaacgym
import torch
import os
from isaacgym.torch_utils import *
from isaacgym.torch_utils import quat_from_euler_xyz, quat_mul

# 直接运行 scripts/play_by_key.py 时，sys.path 只有 scripts/，仓库根目录不在其中
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from go1_gym.envs import *
from go1_gym.envs.automatic import KeyboardWrapper
from go1_gym.utils import quaternion_to_rpy
from scripts.load_policy import load_dog_policy, load_arm_policy, load_unified_policy, load_env
from scripts.play_eval_once import EvalOnceRunner
import argparse

logdir = "runs/test_go212.11/181025.391800_seed1133"
ckpt_id = "040000"

x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
roll_cmd, pitch_cmd, yaw_cmd = 0.1, 0.5, 0.0


def _new_joint_range_tracker(device, num_joints):
    inf = float("inf")
    return {
        "q_min": torch.full((num_joints,), inf, device=device),
        "q_max": torch.full((num_joints,), -inf, device=device),
    }


def _update_joint_range_tracker(tracker, q_now):
    tracker["q_min"] = torch.minimum(tracker["q_min"], q_now)
    tracker["q_max"] = torch.maximum(tracker["q_max"], q_now)


def _detect_outliers_by_mad(values, z_thresh=2.5, eps=1e-8):
    median = torch.median(values)
    mad = torch.median(torch.abs(values - median))
    if mad.item() < eps:
        zeros = torch.zeros_like(values)
        return (
            torch.zeros_like(values, dtype=torch.bool),
            torch.zeros_like(values, dtype=torch.bool),
            zeros,
        )
    rz = 0.6745 * (values - median) / (mad + eps)
    return rz > z_thresh, rz < -z_thresh, rz


def _build_loco_print_order(dof_names, num_loco_joints):
    """Order: hip across legs, then thigh, then calf (FL/FR/RL/RR)."""
    if dof_names is None:
        return list(range(num_loco_joints))
    legs = ("FL", "FR", "RL", "RR")
    joint_types = ("hip_joint", "thigh_joint", "calf_joint")
    name_to_idx = {n: i for i, n in enumerate(dof_names[:num_loco_joints])}
    order = []
    for jt in joint_types:
        for leg in legs:
            idx = name_to_idx.get(f"{leg}_{jt}", None)
            if idx is not None:
                order.append(idx)
    if len(order) != num_loco_joints:
        return list(range(num_loco_joints))
    return order


def _wrap_to_pi(angles):
    return torch.atan2(torch.sin(angles), torch.cos(angles))


def _get_arm_hold_mask(base_env, args, last_cmd_obs, arm_hold_active):
    env_ids = torch.arange(base_env.num_envs, device=base_env.device)
    cmd_obs = base_env.commands_arm_obs[:, :6]
    lpy = base_env.get_lpy_in_base_coord(env_ids)
    abg = base_env.get_alpha_beta_gamma_in_base_coord(env_ids)
    lpy_err = torch.norm(lpy - cmd_obs[:, :3], dim=1)
    rpy_err = torch.norm(_wrap_to_pi(abg - base_env.target_abg), dim=1)

    target_changed = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
    if last_cmd_obs is not None:
        target_changed = torch.norm(cmd_obs - last_cmd_obs, dim=1) > args.arm_hold_cmd_change_threshold

    near_target = (
        (lpy_err < args.arm_hold_lpy_threshold)
        & (rpy_err < args.arm_hold_rpy_threshold)
    )
    if args.arm_hold_use_vel:
        arm_start = base_env.num_actions_loco
        arm_end = arm_start + base_env.num_actions_arm
        arm_vel = torch.max(torch.abs(base_env.dof_vel[:, arm_start:arm_end]), dim=1).values
        near_target &= arm_vel < args.arm_hold_vel_threshold

    return (arm_hold_active | near_target) & ~target_changed


def _restore_fixed_ee_goal(env, fixed_goal_lpy, fixed_goal_abg, env_ids=None):
    """Restore fixed EE command (LPY+ABG) and sync keyboard XYZ target from it."""
    base_env = env.env if hasattr(env, "env") else env
    if env_ids is None:
        env.commands_arm[:, 0:3] = fixed_goal_lpy.unsqueeze(0)
        env.commands_arm[:, 3:6] = fixed_goal_abg.unsqueeze(0)
    elif len(env_ids) > 0:
        env.commands_arm[env_ids, 0:3] = fixed_goal_lpy.unsqueeze(0)
        env.commands_arm[env_ids, 3:6] = fixed_goal_abg.unsqueeze(0)
    if hasattr(env, "update_arm_commands"):
        env.update_arm_commands()
    elif hasattr(base_env, "update_arm_commands"):
        base_env.update_arm_commands()
    if getattr(base_env, "use_xyz_arm_keyboard", False):
        base_env._sync_arm_target_xyz_from_commands()


def _foot_names_in_order(base_env):
    foot_token = base_env.cfg.asset.foot_name
    return [s for s in base_env.body_names if foot_token in s]


def _vec_changed(new_tuple, old_tuple, eps):
    if old_tuple is None:
        return False
    if len(new_tuple) != len(old_tuple):
        return True
    return any(abs(a - b) > eps for a, b in zip(new_tuple, old_tuple))


def _maybe_print_physics_debug(base_env, args, sim_step, sim_t, state):
    """Print foot contact forces and/or body-frame projected gravity on a timer."""
    want_feet = args.print_foot_force
    want_pg = args.print_projected_gravity
    if not want_feet and not want_pg:
        return
    if sim_t < state["next_check_t"]:
        return

    state["next_check_t"] = sim_t + float(args.print_foot_force_interval)
    foot_eps = float(args.print_foot_force_eps)
    pg_eps = float(args.print_projected_gravity_eps)

    pg_tuple = None
    fz_tuple = None
    foot_cf = None
    foot_fn = None
    foot_names = None

    if want_pg and hasattr(base_env, "projected_gravity"):
        pg_tuple = tuple(float(x) for x in base_env.projected_gravity[0].cpu().tolist())

    if want_feet and hasattr(base_env, "contact_forces") and hasattr(base_env, "feet_indices"):
        foot_cf = base_env.contact_forces[0, base_env.feet_indices, :]
        foot_fn = torch.norm(foot_cf, dim=-1)
        foot_names = _foot_names_in_order(base_env)
        fz_tuple = tuple(float(x) for x in foot_cf[:, 2].cpu().tolist())

    if not state.get("warmup_done", False):
        if pg_tuple is not None:
            state["last_pg"] = pg_tuple
        if fz_tuple is not None:
            state["last_fz"] = fz_tuple
        state["warmup_done"] = True
        return

    pg_changed = pg_tuple is not None and _vec_changed(
        pg_tuple, state.get("last_pg"), pg_eps
    )
    foot_changed = fz_tuple is not None and _vec_changed(
        fz_tuple, state.get("last_fz"), foot_eps
    )
    if pg_tuple is not None:
        state["last_pg"] = pg_tuple
    if fz_tuple is not None:
        state["last_fz"] = fz_tuple

    if not pg_changed and not foot_changed:
        return

    chunks = [f"[PHYSICS] t={sim_t:.2f}s step={sim_step}"]
    if pg_tuple is not None:
        gnorm = math.sqrt(sum(x * x for x in pg_tuple))
        chunks.append(
            f"proj_gravity_body=[{pg_tuple[0]:.4f},{pg_tuple[1]:.4f},{pg_tuple[2]:.4f}] "
            f"|g|={gnorm:.4f}"
        )
    if fz_tuple is not None and foot_cf is not None:
        foot_parts = [f"contact(z>1N)={(foot_cf[:, 2] > 1.0).tolist()}"]
        for i in range(foot_cf.shape[0]):
            name = foot_names[i] if foot_names and i < len(foot_names) else f"foot_{i}"
            fx, fy, fzi = foot_cf[i, 0].item(), foot_cf[i, 1].item(), foot_cf[i, 2].item()
            foot_parts.append(
                f"{name}: F_world=[{fx:.2f},{fy:.2f},{fzi:.2f}]N "
                f"fz={fzi:.2f}N |F|={foot_fn[i].item():.2f}N"
            )
        chunks.append(" | ".join(foot_parts))
    print(" | ".join(chunks))


def play_go1(args):
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd, roll_cmd, pitch_cmd, yaw_cmd, logdir, ckpt_id
    
    logdir = args.logdir
    ckpt_id = str(args.ckptid).zfill(6)
            
    from go1_gym.utils.global_switch import global_switch
    global_switch.open_switch()
    
    env, cfg = load_env(logdir, wrapper=KeyboardWrapper, headless=args.headless, device=args.sim_device)
    base_env = env.env if hasattr(env, "env") else env
    base_env.dump_action_npz = bool(args.dump_action_npz)
    base_env.dump_action_npz_path = args.dump_action_npz_path
    if base_env.dump_action_npz:
        if not base_env.dump_action_npz_path:
            base_env.dump_action_npz_path = os.path.join(logdir, f"action_trace_ckpt{ckpt_id}.npz")
        print(f"[Play Mode] action trace dump enabled: {base_env.dump_action_npz_path}")
    
    unified_policy = None
    if args.unified:
        unified_policy = load_unified_policy(
            logdir,
            ckpt_id,
            cfg,
            ckpt_file=args.unified_ckpt_file,
        )
        dog_policy = None
        arm_policy = None
    else:
        dog_policy = load_dog_policy(
            logdir, ckpt_id, cfg, ckpt_file=args.dog_ckpt_file
        )
        arm_policy = None if args.dog_only else load_arm_policy(
            logdir,
            ckpt_id,
            cfg,
            ckpt_file=args.arm_ckpt_file,
        )
    
    env.env.enable_viewer_sync = True

    num_eval_steps = 100000

    ''' press 'F' to fixed camera'''
    # cam_pos = gymapi.Vec3(4, 3, 2)
    # cam_target = gymapi.Vec3(-4, -3, 0)
    # env.gym.viewer_camera_look_at(env.viewer, env.envs[0], cam_pos, cam_target)

    # Fixed EE goal from user-specified initialization print.
    fixed_goal_lpy = torch.tensor(
        [0.3965710401535034, 0.695070743560791, -0.004516037181019783],
        dtype=torch.float32,
        device=env.device,
    )
    fixed_goal_abg = torch.tensor(
        [-0.0028180349618196487, 0.12667569518089294, -0.008409993723034859],
        dtype=torch.float32,
        device=env.device,
    )

    def _enable_arm_xyz_keyboard():
        base_env.use_xyz_arm_keyboard = True
        _restore_fixed_ee_goal(env, fixed_goal_lpy, fixed_goal_abg)
        base_env._print_arm_xyz_cmd(force=True)

    obs = env.reset()
    _enable_arm_xyz_keyboard()
    evaluator = EvalOnceRunner(num_joints=env.env.num_actions_loco, dog_only=args.dog_only)
    eval_repeats = 3
    eval_batch_active = False
    eval_batch_results = []
    eval_batch_idx = 0
    num_loco_joints = int(base_env.num_actions_loco)
    if hasattr(base_env, "dof_names") and len(base_env.dof_names) >= num_loco_joints:
        range_joint_names = list(base_env.dof_names[:num_loco_joints])
    else:
        range_joint_names = [f"joint_{i}" for i in range(num_loco_joints)]
    eval_batch_range_results = []
    run_range_tracker = None

    env.commands_dog[:, 0] = x_vel_cmd
    env.commands_dog[:, 1] = y_vel_cmd
    env.commands_dog[:, 2] = yaw_vel_cmd

    count = 0
    last_arm_action_step = torch.zeros(
        env.num_envs, env.env.num_actions_arm, device=env.device
    )
    last_arm_plan_step = torch.zeros(env.num_envs, 2, device=env.device)
    has_last_arm_action = False
    last_arm_cmd_obs = None
    arm_hold_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    plan_rp_next_check_t = float(args.print_plan_interval)
    last_plan_rp_cmd_checked = None
    foot_force_state = {
        "next_check_t": float(args.print_foot_force_interval),
        "last_fz": None,
        "last_pg": None,
        "warmup_done": False,
    }

    for i in (range(num_eval_steps)):
        if hasattr(env.env, "eval_trigger") and env.env.eval_trigger and not evaluator.active and not eval_batch_active:
            env.env.eval_trigger = False
            eval_batch_active = True
            eval_batch_results = []
            eval_batch_range_results = []
            eval_batch_idx = 0
            run_range_tracker = None
            env.reset()
            _restore_fixed_ee_goal(env, fixed_goal_lpy, fixed_goal_abg)
            has_last_arm_action = False
            last_arm_cmd_obs = None
            arm_hold_active.zero_()
            plan_rp_next_check_t = float(args.print_plan_interval)
            last_plan_rp_cmd_checked = None
            foot_force_state = {
                "next_check_t": float(args.print_foot_force_interval),
                "last_fz": None,
                "last_pg": None,
                "warmup_done": False,
            }
            evaluator.start()
            print(f"[Eval] started run 1/{eval_repeats}")

        evaluator.apply_command(env)

        with torch.no_grad():
            t1 = time.time()
            actions_arm_step = torch.zeros(
                env.num_envs, env.env.num_actions_arm, device=env.device
            )
            policy_info = {}

            hold_mask = None
            if (
                args.arm_hold
                and has_last_arm_action
                and hasattr(base_env, "commands_arm_obs")
            ):
                arm_hold_active = _get_arm_hold_mask(
                    base_env, args, last_arm_cmd_obs, arm_hold_active
                )
                hold_mask = arm_hold_active

            if args.unified:
                unified_obs = env.get_observations()
                actions_all = unified_policy(unified_obs, info=policy_info)
                actions_all = actions_all.to(env.device)
                actions_dog = actions_all[..., : env.env.num_actions_loco]
                if args.dog_only:
                    actions_arm_step.zero_()
                else:
                    actions_arm_step = actions_all[..., env.env.num_actions_loco :]
                    if hold_mask is not None:
                        actions_arm_step = torch.where(
                            hold_mask.unsqueeze(1), last_arm_action_step, actions_arm_step
                        )
            else:
                if not args.dog_only:
                    obs = env.get_arm_observations()
                    actions_arm = arm_policy(obs, info=policy_info)
                    arm_plan_step = actions_arm[..., -2:].to(env.device)
                    if hold_mask is not None:
                        arm_plan_step = torch.where(
                            hold_mask.unsqueeze(1), last_arm_plan_step, arm_plan_step
                        )
                    env.plan(arm_plan_step)
                    if args.print_plan_rp and actions_arm.shape[-1] >= 2:
                        be = env.env if hasattr(env, "env") else env
                        sim_t = i * float(be.dt)
                        if sim_t >= plan_rp_next_check_t and hasattr(be, "commands_dog"):
                            plan_rp_next_check_t = sim_t + float(args.print_plan_interval)
                            cmd_p = be.commands_dog[0, 3].item()
                            cmd_r = be.commands_dog[0, 4].item()
                            eps = float(args.print_plan_eps)
                            if last_plan_rp_cmd_checked is not None:
                                prev_p, prev_r = last_plan_rp_cmd_checked
                                cmd_changed = (
                                    abs(cmd_p - prev_p) > eps or abs(cmd_r - prev_r) > eps
                                )
                                if cmd_changed:
                                    plan_pitch = actions_arm[0, -2].item()
                                    plan_roll = actions_arm[0, -1].item()
                                    scaled_pitch = 0.4 * plan_pitch
                                    scaled_roll = 0.4 * plan_roll
                                    print(
                                        f"[PLAN-RP] t={sim_t:.2f}s step={i} "
                                        f"raw_pitch={plan_pitch:.4f} raw_roll={plan_roll:.4f} | "
                                        f"scaled_pitch={scaled_pitch:.4f} scaled_roll={scaled_roll:.4f} | "
                                        f"cmd_pitch={cmd_p:.4f} cmd_roll={cmd_r:.4f}"
                                    )
                            last_plan_rp_cmd_checked = (cmd_p, cmd_r)
                    actions_arm_step = actions_arm[..., :-2].to(env.device)
                    if hold_mask is not None:
                        actions_arm_step = torch.where(
                            hold_mask.unsqueeze(1), last_arm_action_step, actions_arm_step
                        )

                dog_obs = env.get_dog_observations()

                actions_dog = dog_policy(dog_obs, info=policy_info)
                actions_dog = actions_dog.to(env.device)

        if args.arm_hold and not args.dog_only:
            last_arm_action_step[:] = actions_arm_step
            if not args.unified and "arm_plan_step" in locals():
                last_arm_plan_step[:] = arm_plan_step
            if hasattr(base_env, "commands_arm_obs"):
                last_arm_cmd_obs = base_env.commands_arm_obs[:, :6].clone()
            has_last_arm_action = True
        ret = env.step(actions_dog, actions_arm_step)
        if args.print_foot_force or args.print_projected_gravity:
            _maybe_print_physics_debug(
                base_env, args, i, i * float(base_env.dt), foot_force_state
            )
        done_buf = ret[2] if len(ret) > 2 else None
        if done_buf is not None:
            reset_env_ids = done_buf.nonzero(as_tuple=False).flatten()
            if len(reset_env_ids) > 0:
                _restore_fixed_ee_goal(env, fixed_goal_lpy, fixed_goal_abg, reset_env_ids)
        if getattr(base_env, "_manual_reset_pending_history_clear", False):
            _restore_fixed_ee_goal(env, fixed_goal_lpy, fixed_goal_abg)

        if args.print_ee_pose and hasattr(base_env, "end_effector_state"):
            every = max(1, int(args.print_ee_every))
            if i % every == 0:
                ee = base_env.end_effector_state[0]
                ee_pos = ee[:3]
                ee_rpy = quaternion_to_rpy(ee[3:7].unsqueeze(0))[0]

                cmd_xyz_str = "N/A"
                cmd_rpy_str = "N/A"
                err_str = "N/A"

                if hasattr(base_env, "commands_arm") and hasattr(base_env, "lpy_to_world_xyz"):
                    x_t, y_t, z_t = base_env.lpy_to_world_xyz()
                    target_pos = torch.stack((x_t, y_t, z_t))
                    cmd_rpy = base_env.commands_arm[0, 3:6]
                    d_pos = torch.norm(ee_pos - target_pos).item()

                    roll = base_env.commands_arm[0:1, 3]
                    pitch = base_env.commands_arm[0:1, 4]
                    yaw = base_env.commands_arm[0:1, 5]
                    zero_vec = torch.zeros_like(roll)
                    q1 = quat_from_euler_xyz(zero_vec, zero_vec, yaw)
                    q2 = quat_from_euler_xyz(zero_vec, pitch, zero_vec)
                    q3 = quat_from_euler_xyz(roll, zero_vec, zero_vec)
                    q_target = quat_mul(q1, quat_mul(q2, q3))[0]
                    cos_half = torch.sum(ee[3:7] * q_target)
                    cos_half = torch.clamp(torch.abs(cos_half), 0.0, 1.0)
                    d_ori = (2.0 * torch.acos(cos_half)).item()
                    d_rpy = torch.abs(ee_rpy - cmd_rpy)
                    cmd_xyz_str = f"[{target_pos[0].item():.4f}, {target_pos[1].item():.4f}, {target_pos[2].item():.4f}]"
                    cmd_rpy_str = f"[{cmd_rpy[0].item():.4f}, {cmd_rpy[1].item():.4f}, {cmd_rpy[2].item():.4f}]"
                    err_str = (
                        f"pos_err_m={d_pos:.4f} "
                        f"rpy_abs_err_rad=[{d_rpy[0].item():.4f}, {d_rpy[1].item():.4f}, {d_rpy[2].item():.4f}] "
                        f"ori_err_rad={d_ori:.4f}"
                    )

                if hasattr(base_env, "commands_arm_obs"):
                    cmd_obs = base_env.commands_arm_obs[0].detach().cpu()
                    raw_cmd = cmd_obs[: min(9, cmd_obs.shape[0])].tolist()
                else:
                    raw_cmd = []
                ee_lpy_str = "N/A"
                if hasattr(base_env, "get_lpy_in_base_coord"):
                    env0 = torch.tensor([0], device=base_env.device)
                    ee_lpy = base_env.get_lpy_in_base_coord(env0)[0]
                    ee_lpy_str = f"[{ee_lpy[0].item():.4f}, {ee_lpy[1].item():.4f}, {ee_lpy[2].item():.4f}]"
                ee_abg_str = "N/A"
                target_abg_str = "N/A"
                abg_err_str = "N/A"
                if (
                    hasattr(base_env, "get_alpha_beta_gamma_in_base_coord")
                    and hasattr(base_env, "target_abg")
                ):
                    env0 = torch.tensor([0], device=base_env.device)
                    ee_abg = base_env.get_alpha_beta_gamma_in_base_coord(env0)[0]
                    target_abg = base_env.target_abg[0]
                    abg_err = torch.abs(_wrap_to_pi(ee_abg - target_abg))
                    ee_abg_str = f"[{ee_abg[0].item():.4f}, {ee_abg[1].item():.4f}, {ee_abg[2].item():.4f}]"
                    target_abg_str = f"[{target_abg[0].item():.4f}, {target_abg[1].item():.4f}, {target_abg[2].item():.4f}]"
                    abg_err_str = f"[{abg_err[0].item():.4f}, {abg_err[1].item():.4f}, {abg_err[2].item():.4f}]"

                print(
                    f"[EE] step={i} xyz=[{ee_pos[0].item():.4f}, {ee_pos[1].item():.4f}, {ee_pos[2].item():.4f}] "
                    f"rpy_rad=[{ee_rpy[0].item():.4f}, {ee_rpy[1].item():.4f}, {ee_rpy[2].item():.4f}]"
                )
                print(f"[CMD] step={i} xyz={cmd_xyz_str} rpy_rad={cmd_rpy_str}")
                print(f"[ERR] step={i} {err_str}")
                print(
                    f"[CMD-OBS] step={i} raw_policy_cmd_target(first_{len(raw_cmd)})={raw_cmd}"
                )
                print(f"[EE-LPY] step={i} lpy={ee_lpy_str}")
                print(f"[EE-ABG] step={i} abg={ee_abg_str}")
                print(f"[TARGET-ABG] step={i} abg={target_abg_str} abs_err={abg_err_str}")

        if eval_batch_active and evaluator.active:
            if run_range_tracker is None:
                run_range_tracker = _new_joint_range_tracker(base_env.device, num_loco_joints)
            q_now = base_env.dof_pos[0, :num_loco_joints]
            _update_joint_range_tracker(run_range_tracker, q_now)

        eval_result = evaluator.update(env)
        if eval_result is not None and eval_batch_active:
            eval_batch_results.append(eval_result)
            if run_range_tracker is not None:
                q_min = run_range_tracker["q_min"].detach().cpu()
                q_max = run_range_tracker["q_max"].detach().cpu()
                q_range = (run_range_tracker["q_max"] - run_range_tracker["q_min"]).detach().cpu()
                eval_batch_range_results.append(
                    {
                        "q_min": q_min,
                        "q_max": q_max,
                        "q_range": q_range,
                    }
                )
            else:
                zero = torch.zeros(num_loco_joints)
                eval_batch_range_results.append(
                    {
                        "q_min": zero.clone(),
                        "q_max": zero.clone(),
                        "q_range": zero.clone(),
                    }
                )
            eval_batch_idx += 1

            if eval_batch_idx < eval_repeats:
                env.reset()
                _restore_fixed_ee_goal(env, fixed_goal_lpy, fixed_goal_abg)
                evaluator.start()
                run_range_tracker = None
                print(f"[Eval] started run {eval_batch_idx + 1}/{eval_repeats}")
            else:
                ate_lin_x = torch.tensor([r["ate_lin_x"] for r in eval_batch_results], dtype=torch.float32)
                ate_lin_y = torch.tensor([r["ate_lin_y"] for r in eval_batch_results], dtype=torch.float32)
                ate_yaw = torch.tensor([r["ate_ang_yaw"] for r in eval_batch_results], dtype=torch.float32)
                joint_err = torch.tensor(
                    [r["tracking_dof_pos_per_joint"] for r in eval_batch_results],
                    dtype=torch.float32
                )

                ate_lin_x_mean = ate_lin_x.mean().item()
                ate_lin_x_std = ate_lin_x.std(unbiased=True).item() if ate_lin_x.numel() > 1 else 0.0
                ate_lin_y_mean = ate_lin_y.mean().item()
                ate_lin_y_std = ate_lin_y.std(unbiased=True).item() if ate_lin_y.numel() > 1 else 0.0
                ate_yaw_mean = ate_yaw.mean().item()
                ate_yaw_std = ate_yaw.std(unbiased=True).item() if ate_yaw.numel() > 1 else 0.0
                joint_mean = joint_err.mean(dim=0)
                joint_std = joint_err.std(dim=0, unbiased=True) if joint_err.shape[0] > 1 else torch.zeros_like(joint_mean)

                print(
                    f"[Eval][Done] N={eval_repeats}, "
                    f"ATE_lin_x={ate_lin_x_mean:.4f}% ± {ate_lin_x_std:.4f}%, "
                    f"ATE_lin_y={ate_lin_y_mean:.4f}% ± {ate_lin_y_std:.4f}%, "
                    f"ATE_ang_yaw={ate_yaw_mean:.4f}% ± {ate_yaw_std:.4f}%"
                )
                survival = torch.tensor([r.get("survival_rate", 0.0) for r in eval_batch_results], dtype=torch.float32)
                survival_mean = survival.mean().item()
                survival_std = survival.std(unbiased=True).item() if survival.numel() > 1 else 0.0
                print(f"[Eval] SurvivalRate={survival_mean:.4f} ± {survival_std:.4f}")

                if not args.dog_only:
                    workspace_xy = torch.tensor([r.get("workspace_area_xy", 0.0) for r in eval_batch_results], dtype=torch.float32)
                    workspace_xz = torch.tensor([r.get("workspace_area_xz", 0.0) for r in eval_batch_results], dtype=torch.float32)
                    workspace_yz = torch.tensor([r.get("workspace_area_yz", 0.0) for r in eval_batch_results], dtype=torch.float32)
                    d_err = torch.tensor([r.get("position_tracking_error_D", 0.0) for r in eval_batch_results], dtype=torch.float32)
                    phi_err = torch.tensor([r.get("orientation_tracking_error_Phi", 0.0) for r in eval_batch_results], dtype=torch.float32)
                    solv = torch.tensor([r.get("solvability", 0.0) for r in eval_batch_results], dtype=torch.float32)
                    lpy_err = torch.tensor([r.get("position_tracking_error_lpy", [0.0, 0.0, 0.0]) for r in eval_batch_results], dtype=torch.float32)
                    abg_err = torch.tensor([r.get("orientation_tracking_error_abg", [0.0, 0.0, 0.0]) for r in eval_batch_results], dtype=torch.float32)

                    def _mean_std_1d(x):
                        mean = x.mean().item()
                        std = x.std(unbiased=True).item() if x.numel() > 1 else 0.0
                        return mean, std

                    wxy_m, wxy_s = _mean_std_1d(workspace_xy)
                    wxz_m, wxz_s = _mean_std_1d(workspace_xz)
                    wyz_m, wyz_s = _mean_std_1d(workspace_yz)
                    d_m, d_s = _mean_std_1d(d_err)
                    phi_m, phi_s = _mean_std_1d(phi_err)
                    solv_m, solv_s = _mean_std_1d(solv)

                    lpy_mean = lpy_err.mean(dim=0)
                    lpy_std = lpy_err.std(dim=0, unbiased=True) if lpy_err.shape[0] > 1 else torch.zeros_like(lpy_mean)
                    abg_mean = abg_err.mean(dim=0)
                    abg_std = abg_err.std(dim=0, unbiased=True) if abg_err.shape[0] > 1 else torch.zeros_like(abg_mean)

                    print(
                        f"[Eval] Workspace area XY={wxy_m:.4f} ± {wxy_s:.4f}, "
                        f"XZ={wxz_m:.4f} ± {wxz_s:.4f}, "
                        f"YZ={wyz_m:.4f} ± {wyz_s:.4f}"
                    )
                    print(
                        f"[Eval] PositionError D={d_m:.4f} ± {d_s:.4f}, "
                        f"OrientationError Phi={phi_m:.4f} ± {phi_s:.4f}, "
                        f"Solvability={solv_m:.4f} ± {solv_s:.4f}"
                    )
                    print(
                        "[Eval] PositionError l,p,y="
                        f"{lpy_mean[0].item():.4f}±{lpy_std[0].item():.4f}, "
                        f"{lpy_mean[1].item():.4f}±{lpy_std[1].item():.4f}, "
                        f"{lpy_mean[2].item():.4f}±{lpy_std[2].item():.4f}"
                    )
                    print(
                        "[Eval] OrientationError alpha,beta,gamma="
                        f"{abg_mean[0].item():.4f}±{abg_std[0].item():.4f}, "
                        f"{abg_mean[1].item():.4f}±{abg_std[1].item():.4f}, "
                        f"{abg_mean[2].item():.4f}±{abg_std[2].item():.4f}"
                    )

                    # Stratified statistics (L/P/Y buckets)
                    def _print_strata(name, key):
                        strata_runs = [r.get(key, []) for r in eval_batch_results]
                        if not strata_runs or not strata_runs[0]:
                            return
                        n_bins = len(strata_runs[0])
                        print(f"[Eval] Stratified {name} bins:")
                        for i in range(n_bins):
                            lows = [s[i]["low"] for s in strata_runs]
                            highs = [s[i]["high"] for s in strata_runs]
                            counts = [s[i]["count"] for s in strata_runs]
                            solv_vals = [s[i]["solvability"] for s in strata_runs]
                            d_vals = [s[i]["d"] for s in strata_runs]
                            phi_vals = [s[i]["phi"] for s in strata_runs]
                            count_mean = sum(counts) / max(1, len(counts))
                            solv_mean = sum(solv_vals) / max(1, len(solv_vals))
                            d_mean_bin = sum(d_vals) / max(1, len(d_vals))
                            phi_mean_bin = sum(phi_vals) / max(1, len(phi_vals))
                            print(
                                f"  [{lows[0]:.3f}, {highs[0]:.3f}]: "
                                f"count={count_mean:.2f}, solv={solv_mean:.4f}, "
                                f"D={d_mean_bin:.4f}, Phi={phi_mean_bin:.4f}"
                            )

                    _print_strata("L", "strata_l")
                    _print_strata("P", "strata_p")
                    _print_strata("Y", "strata_y")

                    # Convergence curve plotting (always on)
                    conv_n_runs = [r.get("conv_curve_n", []) for r in eval_batch_results]
                    if conv_n_runs and conv_n_runs[0]:
                        min_len = min(len(x) for x in conv_n_runs)
                        conv_n = torch.tensor(conv_n_runs[0][:min_len], dtype=torch.float32)
                        conv_xy = torch.tensor(
                            [r.get("conv_curve_workspace_xy", [0.0] * min_len)[:min_len] for r in eval_batch_results],
                            dtype=torch.float32
                        )
                        conv_xz = torch.tensor(
                            [r.get("conv_curve_workspace_xz", [0.0] * min_len)[:min_len] for r in eval_batch_results],
                            dtype=torch.float32
                        )
                        conv_yz = torch.tensor(
                            [r.get("conv_curve_workspace_yz", [0.0] * min_len)[:min_len] for r in eval_batch_results],
                            dtype=torch.float32
                        )
                        conv_solv = torch.tensor(
                            [r.get("conv_curve_solvability", [0.0] * min_len)[:min_len] for r in eval_batch_results],
                            dtype=torch.float32
                        )

                        xy_mean = conv_xy.mean(dim=0)
                        xz_mean = conv_xz.mean(dim=0)
                        yz_mean = conv_yz.mean(dim=0)
                        solv_mean_curve = conv_solv.mean(dim=0)
                        xy_std = conv_xy.std(dim=0, unbiased=True) if conv_xy.shape[0] > 1 else torch.zeros_like(xy_mean)
                        xz_std = conv_xz.std(dim=0, unbiased=True) if conv_xz.shape[0] > 1 else torch.zeros_like(xz_mean)
                        yz_std = conv_yz.std(dim=0, unbiased=True) if conv_yz.shape[0] > 1 else torch.zeros_like(yz_mean)
                        solv_std_curve = conv_solv.std(dim=0, unbiased=True) if conv_solv.shape[0] > 1 else torch.zeros_like(solv_mean_curve)

                        try:
                            import matplotlib.pyplot as plt

                            plot_dir = os.path.join(logdir, "eval_plots")
                            os.makedirs(plot_dir, exist_ok=True)
                            plot_path = os.path.join(plot_dir, f"convergence_ckpt{ckpt_id}_run{int(time.time())}.png")

                            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                            n_np = conv_n.cpu().numpy()
                            axes[0].plot(n_np, xy_mean.cpu().numpy(), label="XY")
                            axes[0].plot(n_np, xz_mean.cpu().numpy(), label="XZ")
                            axes[0].plot(n_np, yz_mean.cpu().numpy(), label="YZ")
                            axes[0].fill_between(
                                n_np,
                                (xy_mean - xy_std).cpu().numpy(),
                                (xy_mean + xy_std).cpu().numpy(),
                                alpha=0.15
                            )
                            axes[0].set_title("Workspace Convergence")
                            axes[0].set_xlabel("Target Samples")
                            axes[0].set_ylabel("Area")
                            axes[0].legend()

                            axes[1].plot(n_np, solv_mean_curve.cpu().numpy(), label="Solvability")
                            axes[1].fill_between(
                                n_np,
                                (solv_mean_curve - solv_std_curve).cpu().numpy(),
                                (solv_mean_curve + solv_std_curve).cpu().numpy(),
                                alpha=0.2
                            )
                            axes[1].set_title("Solvability Convergence")
                            axes[1].set_xlabel("Target Samples")
                            axes[1].set_ylabel("Ratio")
                            axes[1].set_ylim(0.0, 1.0)
                            axes[1].legend()

                            plt.tight_layout()
                            plt.savefig(plot_path, dpi=150)
                            plt.close(fig)
                            print(f"[Eval] Convergence plot saved: {plot_path}")
                        except Exception as e:
                            print(f"[Eval] Convergence plot failed: {e}")
                else:
                    print("[Eval] dog_only=True, skip arm-related metrics.")
                print("[Eval] per-joint tracking_dof_pos (mean ± std):")
                dof_names = env.env.dof_names[:eval_result["num_joints"]]
                print_order = _build_loco_print_order(dof_names, eval_result["num_joints"])
                for j in print_order:
                    print(f"  {dof_names[j]}: {joint_mean[j].item():.4f} ± {joint_std[j].item():.4f}")
                def _mean_std(x):
                    mean = x.mean(dim=0)
                    std = x.std(dim=0, unbiased=True) if x.shape[0] > 1 else torch.zeros_like(mean)
                    return mean, std

                q_min_all = torch.stack([r["q_min"] for r in eval_batch_range_results], dim=0)
                q_max_all = torch.stack([r["q_max"] for r in eval_batch_range_results], dim=0)
                q_range_all = torch.stack([r["q_range"] for r in eval_batch_range_results], dim=0)

                q_min_mean, q_min_std = _mean_std(q_min_all)
                q_max_mean, q_max_std = _mean_std(q_max_all)
                q_range_mean, q_range_std = _mean_std(q_range_all)

                qmin_hi, qmin_lo, _ = _detect_outliers_by_mad(q_min_mean)
                qmax_hi, qmax_lo, _ = _detect_outliers_by_mad(q_max_mean)
                qrng_hi, qrng_lo, _ = _detect_outliers_by_mad(q_range_mean)

                # print("[Eval] per-joint loco q min/max/range (mean ± std), with outlier tags:")
                # print("  tags: QMIN_HI/LO, QMAX_HI/LO, QRNG_HI/LO")
                # for idx in print_order:
                #     joint_name = range_joint_names[idx]
                #     tags = []
                #     if qmin_hi[idx]:
                #         tags.append("QMIN_HI")
                #     if qmin_lo[idx]:
                #         tags.append("QMIN_LO")
                #     if qmax_hi[idx]:
                #         tags.append("QMAX_HI")
                #     if qmax_lo[idx]:
                #         tags.append("QMAX_LO")
                #     if qrng_hi[idx]:
                #         tags.append("QRNG_HI")
                #     if qrng_lo[idx]:
                #         tags.append("QRNG_LO")
                #     tag_str = " | OUTLIER: " + ",".join(tags) if tags else ""
                #
                #     print(
                #         f"  {joint_name}: "
                #         f"q_min={q_min_mean[idx]:.4f} ± {q_min_std[idx]:.4f}, "
                #         f"q_max={q_max_mean[idx]:.4f} ± {q_max_std[idx]:.4f}, "
                #         f"q_range={q_range_mean[idx]:.4f} ± {q_range_std[idx]:.4f}"
                #         f"{tag_str}"
                #     )
                print("[Eval] waiting next T")
                eval_batch_active = False



if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    parser = argparse.ArgumentParser(description="Go1")
    parser.add_argument('--headless', action='store_true', default=False)
    parser.add_argument('--sim_device', type=str, default="cuda:0")
    parser.add_argument('--robot', type=str, default="go1", choices=["go1", "go2", "go2_airbot", "go2_piper"],
                       help='Robot type hint for CLI compatibility (play env still follows training config in logdir).')
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--ckptid', type=int, default=40000)
    parser.add_argument('--dog_ckpt_file', type=str, default=None,
                       help='Full path to dog checkpoint file (.pt). Higher priority than --ckptid.')
    parser.add_argument('--arm_ckpt_file', type=str, default=None,
                       help='Full path to arm checkpoint file (.pt). Higher priority than --ckptid.')
    parser.add_argument('--unified_ckpt_file', type=str, default=None,
                       help='Full path to unified checkpoint file (.pt). Higher priority than --ckptid.')
    parser.add_argument('--dog_only', action='store_true', default=False,
                       help='Play with dog policy only (skip loading and running arm policy).')
    parser.add_argument('--unified', action='store_true', default=False,
                       help='Load unified policy from checkpoints_unified instead of dog/arm split checkpoints.')
    parser.add_argument('--dump_action_npz', action='store_true', default=False,
                       help='Print and dump action traces (raw_action/delta/target) to npz during play.')
    parser.add_argument('--dump_action_npz_path', type=str, default=None,
                       help='Output path for action trace npz. Default: <logdir>/action_trace_ckptXXXXXX.npz')
    parser.add_argument('--print_ee_pose', action='store_true', default=False,
                       help='Print end-effector world pose (position, quat xyzw, RPY rad) during play.')
    parser.add_argument('--print_ee_every', type=int, default=10,
                       help='Print EE pose every N sim steps when --print_ee_pose is set (default 10).')
    parser.add_argument('--print_plan_rp', action='store_true', default=False,
                       help='Every --print_plan_interval seconds, print plan RP only if cmd_pitch/cmd_roll changed.')
    parser.add_argument('--print_plan_interval', type=float, default=1.0,
                       help='Seconds between cmd_pitch/cmd_roll checks when --print_plan_rp is set.')
    parser.add_argument('--print_plan_eps', type=float, default=1e-4,
                       help='Min abs delta on cmd_pitch/cmd_roll to count as changed.')
    parser.add_argument('--print_foot_force', action='store_true', default=False,
                       help='Every --print_foot_force_interval s, print per-foot contact forces if changed.')
    parser.add_argument('--print_projected_gravity', action='store_true', default=False,
                       help='Every --print_foot_force_interval s, print body-frame projected_gravity if changed.')
    parser.add_argument('--print_foot_force_interval', type=float, default=1.0,
                       help='Seconds between physics debug checks (--print_foot_force / --print_projected_gravity).')
    parser.add_argument('--print_foot_force_eps', type=float, default=1.0,
                       help='Min abs delta on each foot fz (N) to count as changed.')
    parser.add_argument('--print_projected_gravity_eps', type=float, default=0.01,
                       help='Min abs delta on each projected_gravity component to count as changed.')
    parser.add_argument('--disable_arm_hold', dest='arm_hold', action='store_false',
                       help='Disable play-time arm hold near the EE goal.')
    parser.set_defaults(arm_hold=True)
    parser.add_argument('--arm_hold_lpy_threshold', type=float, default=0.05,
                       help='Hold arm action when LPY error norm is below this threshold.')
    parser.add_argument('--arm_hold_rpy_threshold', type=float, default=0.3,
                       help='Hold arm action when orientation error norm is below this threshold.')
    parser.add_argument('--arm_hold_vel_threshold', type=float, default=0.08,
                       help='Max arm joint velocity threshold used only with --arm_hold_use_vel.')
    parser.add_argument('--arm_hold_use_vel', action='store_true', default=False,
                       help='Require low arm joint velocity before entering hold.')
    parser.add_argument('--arm_hold_cmd_change_threshold', type=float, default=1e-4,
                       help='Release arm hold when command observation changes by more than this norm.')
   
    args = parser.parse_args()
    
    if args.dog_only and args.arm_ckpt_file is not None:
        print("[Warning] --arm_ckpt_file is ignored when --dog_only is enabled.")
    if args.unified and args.dog_ckpt_file is not None:
        print("[Warning] --dog_ckpt_file is ignored when --unified is enabled.")
    if args.unified and args.arm_ckpt_file is not None:
        print("[Warning] --arm_ckpt_file is ignored when --unified is enabled.")
    if args.robot != "go1":
        print(
            f"[Info] --robot={args.robot} received. "
            "play_by_key currently loads robot setup from logdir/parameters.pkl."
        )
    
    play_go1(args)

