import math
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_conjugate, quat_mul


def _convex_hull_area_2d(points_2d):
    if len(points_2d) < 3:
        return 0.0
    pts = sorted(set((float(x), float(y)) for x, y in points_2d))
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0

    area = 0.0
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % len(hull)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


class EvalOnceRunner:
    """Single-run play evaluator triggered by keyboard event."""

    def __init__(self, num_joints=None, dog_only=False, sobol_seed=42):
        # (lin_x, lin_y, yaw_z, duration_steps)
        self.schedule = [
            (0.30, 0.00, 0.00, 300),
            (0.60, 0.00, 0.00, 300),
            (0.00, 0.30, 0.00, 300),
            (0.00, -0.30, 0.00, 300),
            (0.50, 0.20, 0.40, 300),
            (0.50, -0.20, -0.40, 300),
            (0.00, 0.00, 0.80, 300),
            (0.00, 0.00, -0.80, 300),
            (0.00, 0.00, 0.00, 200),
        ]
        self.total_steps = sum(seg[3] for seg in self.schedule)
        self.fixed_num_joints = num_joints
        self.dog_only = dog_only
        self.sobol_seed = sobol_seed
        self.active = False
        self.step_idx = 0
        self._reset_stats()

    def _reset_stats(self):
        self.num_steps = 0
        self.sum_pct_err_lin_x = 0.0
        self.sum_pct_err_lin_y = 0.0
        self.sum_pct_err_yaw_z = 0.0
        self.num_valid_lin_x = 0
        self.num_valid_lin_y = 0
        self.num_valid_yaw_z = 0
        self.cmd_eps = 1e-6
        self.sum_abs_dof_pos_per_joint = None
        self.num_joints = self.fixed_num_joints

        # Arm-related metrics
        self.pos_err_sum = 0.0
        self.ori_err_sum = 0.0
        self.lpy_err_sum = torch.zeros(3, dtype=torch.float32)
        self.rpy_err_sum = torch.zeros(3, dtype=torch.float32)
        self.arm_samples = 0
        self.solved_samples = 0
        self.workspace_points = []
        self.pos_thr = 0.03
        self.ori_thr = math.pi / 36.0
        self.target_records = []
        self.steady_tail_len = 20

        # Survival metric with disturbance
        self.min_base_height = float("inf")
        self.survival_height_thr = 0.26
        self.disturbance_initialized = False
        self.disturbance_start = 0
        self.disturbance_end = 0
        self.disturbance_force_vec = None
        self.disturbance_mass = 12.0

        # Fixed Sobol targets
        self.arm_target_resample_steps = 120
        self.max_target_count = max(1, self.total_steps // self.arm_target_resample_steps + 1)
        self.sobol_targets = None
        self.target_idx = -1
        self.current_target = None
        self._build_sobol_targets()

    def _build_sobol_targets(self):
        if self.dog_only:
            self.sobol_targets = None
            return
        engine = torch.quasirandom.SobolEngine(dimension=6, scramble=True, seed=self.sobol_seed)
        u = engine.draw(self.max_target_count)
        low = torch.tensor([0.2, 0.2, -0.6, -0.4, -0.4, -0.4], dtype=torch.float32)
        high = torch.tensor([0.8, 0.8, 0.6, 0.4, 0.4, 0.4], dtype=torch.float32)
        self.sobol_targets = low + (high - low) * u

    def start(self):
        self.active = True
        self.step_idx = 0
        self._reset_stats()

    def _current_command(self):
        cursor = self.step_idx
        acc = 0
        for lin_x, lin_y, yaw_z, dur in self.schedule:
            if cursor < acc + dur:
                return lin_x, lin_y, yaw_z
            acc += dur
        return 0.0, 0.0, 0.0

    def _maybe_init_disturbance(self, base_env):
        if self.disturbance_initialized:
            return
        self.disturbance_initialized = True
        dt = float(base_env.dt) if hasattr(base_env, "dt") else 0.02
        dur_steps = max(1, int(round(2.0 / dt)))
        self.disturbance_start = max(0, self.total_steps // 3)
        self.disturbance_end = min(self.total_steps, self.disturbance_start + dur_steps)
        mag = 10.0 + 10.0 * float(torch.rand(1).item())
        direction = torch.randn(3)
        direction = direction / (torch.norm(direction) + 1e-8)
        self.disturbance_force_vec = direction * mag

    def _apply_base_disturbance(self, base_env):
        if self.disturbance_force_vec is None or not hasattr(base_env, "root_states"):
            return
        dt = float(base_env.dt) if hasattr(base_env, "dt") else 0.02
        dv = (self.disturbance_force_vec.to(base_env.device) / self.disturbance_mass) * dt
        base_env.root_states[0, 7:10] += dv
        base_env.gym.set_actor_root_state_tensor(base_env.sim, gymtorch.unwrap_tensor(base_env.root_states))

    def _target_xyz_world(self, base_env):
        x, y, z = base_env.lpy_to_world_xyz()
        return float(x.item()), float(y.item()), float(z.item())

    def _target_quat_world(self, base_env):
        roll = float(base_env.commands_arm[0, 3].item())
        pitch = float(base_env.commands_arm[0, 4].item())
        yaw = float(base_env.commands_arm[0, 5].item())
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        qw = cr * cp * cy + sr * sp * sy
        return torch.tensor([qx, qy, qz, qw], device=base_env.device, dtype=torch.float32)

    def _quat_geodesic(self, q1, q2):
        dot = torch.sum(q1 * q2, dim=-1).abs().clamp(max=1.0)
        return 2.0 * torch.acos(dot)

    def _collision_free_proxy(self, base_env):
        if not hasattr(base_env, "contact_forces") or not hasattr(base_env, "penalised_contact_indices"):
            return True
        idx = base_env.penalised_contact_indices
        if idx.numel() == 0:
            return True
        forces = base_env.contact_forces[0, idx, :]
        return not bool((torch.norm(forces, dim=-1) > 1.0).any().item())

    def _set_sobol_target(self, base_env, target_idx):
        if self.sobol_targets is None:
            return
        tid = int(max(0, min(target_idx, self.sobol_targets.shape[0] - 1)))
        target = self.sobol_targets[tid].to(base_env.device)
        base_env.commands_arm[0, 0:6] = target

        tx, ty, tz = self._target_xyz_world(base_env)
        tquat = self._target_quat_world(base_env)
        self.current_target = {
            "idx": tid,
            "xyz": (tx, ty, tz),
            "quat": tquat,
            "d_tail": [],
            "phi_tail": [],
            "lpy_tail": [],
            "abg_tail": [],
            "collision_free_all": True,
            "steps": 0,
            "l_val": float(target[0].item()),
            "p_val": float(target[1].item()),
            "y_val": float(target[2].item()),
        }

    def _finalize_current_target(self):
        if self.current_target is None:
            return
        steps = max(1, int(self.current_target["steps"]))

        # Use steady-state tail window to avoid transient bias right after target switch.
        if len(self.current_target["d_tail"]) > 0:
            d_mean = sum(self.current_target["d_tail"]) / len(self.current_target["d_tail"])
        else:
            d_mean = 0.0
        if len(self.current_target["phi_tail"]) > 0:
            phi_mean = sum(self.current_target["phi_tail"]) / len(self.current_target["phi_tail"])
        else:
            phi_mean = 0.0

        if len(self.current_target["lpy_tail"]) > 0:
            lpy_mean = torch.stack(self.current_target["lpy_tail"], dim=0).mean(dim=0)
        else:
            lpy_mean = torch.zeros(3, dtype=torch.float32)
        if len(self.current_target["abg_tail"]) > 0:
            abg_mean = torch.stack(self.current_target["abg_tail"], dim=0).mean(dim=0)
        else:
            abg_mean = torch.zeros(3, dtype=torch.float32)

        solved = (
            d_mean <= self.pos_thr
            and phi_mean <= self.ori_thr
            and bool(self.current_target["collision_free_all"])
        )
        if solved:
            self.workspace_points.append(self.current_target["xyz"])
            self.solved_samples += 1

        self.pos_err_sum += d_mean
        self.ori_err_sum += phi_mean
        self.lpy_err_sum += lpy_mean
        self.rpy_err_sum += abg_mean
        self.arm_samples += 1

        self.target_records.append(
            {
                "xyz": self.current_target["xyz"],
                "d": d_mean,
                "phi": phi_mean,
                "solved": solved,
                "l": self.current_target["l_val"],
                "p": self.current_target["p_val"],
                "y": self.current_target["y_val"],
            }
        )
        self.current_target = None

    def _build_stratified_stats(self, values, records):
        stats = []
        for b0, b1 in zip(values[:-1], values[1:]):
            bucket = [r for r in records if b0 <= r["v"] < b1 or (b1 == values[-1] and r["v"] <= b1)]
            if len(bucket) == 0:
                stats.append({"low": b0, "high": b1, "count": 0, "solvability": 0.0, "d": 0.0, "phi": 0.0})
                continue
            solv = sum(1 for r in bucket if r["solved"]) / len(bucket)
            d_mean = sum(r["d"] for r in bucket) / len(bucket)
            phi_mean = sum(r["phi"] for r in bucket) / len(bucket)
            stats.append({"low": b0, "high": b1, "count": len(bucket), "solvability": solv, "d": d_mean, "phi": phi_mean})
        return stats

    @torch.no_grad()
    def apply_command(self, env):
        if not self.active:
            return
        base_env = env.env if hasattr(env, "env") else env
        lin_x, lin_y, yaw_z = self._current_command()
        base_env.commands_dog[:, 0] = lin_x
        base_env.commands_dog[:, 1] = lin_y
        base_env.commands_dog[:, 2] = yaw_z

        if self.dog_only:
            return
        if self.step_idx % self.arm_target_resample_steps == 0:
            self._finalize_current_target()
            self.target_idx += 1
            self._set_sobol_target(base_env, self.target_idx)

    @torch.no_grad()
    def update(self, env):
        if not self.active:
            return None

        base_env = env.env if hasattr(env, "env") else env
        self._maybe_init_disturbance(base_env)
        if self.disturbance_start <= self.step_idx < self.disturbance_end:
            self._apply_base_disturbance(base_env)

        cmd_lin_x = float(base_env.commands_dog[0, 0].item())
        cmd_lin_y = float(base_env.commands_dog[0, 1].item())
        cmd_yaw_z = float(base_env.commands_dog[0, 2].item())
        act_lin_x = float(base_env.base_lin_vel[0, 0].item())
        act_lin_y = float(base_env.base_lin_vel[0, 1].item())
        act_yaw_z = float(base_env.base_ang_vel[0, 2].item())

        abs_err_lin_x = abs(act_lin_x - cmd_lin_x)
        abs_err_lin_y = abs(act_lin_y - cmd_lin_y)
        abs_err_yaw_z = abs(act_yaw_z - cmd_yaw_z)
        if abs(cmd_lin_x) > self.cmd_eps:
            self.sum_pct_err_lin_x += (abs_err_lin_x / abs(cmd_lin_x)) * 100.0
            self.num_valid_lin_x += 1
        if abs(cmd_lin_y) > self.cmd_eps:
            self.sum_pct_err_lin_y += (abs_err_lin_y / abs(cmd_lin_y)) * 100.0
            self.num_valid_lin_y += 1
        if abs(cmd_yaw_z) > self.cmd_eps:
            self.sum_pct_err_yaw_z += (abs_err_yaw_z / abs(cmd_yaw_z)) * 100.0
            self.num_valid_yaw_z += 1

        if self.num_joints is None:
            self.num_joints = int(base_env.num_actions_loco)
        if self.sum_abs_dof_pos_per_joint is None:
            self.sum_abs_dof_pos_per_joint = torch.zeros(self.num_joints, device=base_env.device)
        target = base_env.joint_pos_target[0, :self.num_joints]
        current = base_env.dof_pos[0, :self.num_joints]
        self.sum_abs_dof_pos_per_joint += torch.abs(target - current)
        self.min_base_height = min(self.min_base_height, float(base_env.base_pos[0, 2].item()))

        if not self.dog_only and self.current_target is not None:
            target_pos = torch.tensor(self.current_target["xyz"], device=base_env.device, dtype=torch.float32)
            actual_pos = base_env.end_effector_state[0, :3]
            d_pos = float(torch.norm(actual_pos - target_pos).item())
            q_target = self.current_target["quat"].unsqueeze(0)
            q_actual = base_env.end_effector_state[0:1, 3:7]
            d_ori = float(self._quat_geodesic(q_actual, q_target).item())

            self.current_target["d_tail"].append(d_pos)
            self.current_target["phi_tail"].append(d_ori)
            if len(self.current_target["d_tail"]) > self.steady_tail_len:
                self.current_target["d_tail"].pop(0)
            if len(self.current_target["phi_tail"]) > self.steady_tail_len:
                self.current_target["phi_tail"].pop(0)
            self.current_target["collision_free_all"] = self.current_target["collision_free_all"] and self._collision_free_proxy(base_env)
            self.current_target["steps"] += 1

            if hasattr(base_env, "get_lpy_in_base_coord"):
                lpy_actual = base_env.get_lpy_in_base_coord(torch.tensor([0], device=base_env.device))[0]
                lpy_target = base_env.commands_arm[0, 0:3]
                self.current_target["lpy_tail"].append(torch.abs(lpy_actual - lpy_target).detach().cpu())
                if len(self.current_target["lpy_tail"]) > self.steady_tail_len:
                    self.current_target["lpy_tail"].pop(0)
            q_base_inv = quat_conjugate(base_env.base_quat[0:1, :])
            q_ee_base = quat_mul(q_base_inv, q_actual)
            abg_actual = base_env.quat_to_angle(q_ee_base)[0]
            abg_target = base_env.commands_arm[0, 3:6]
            self.current_target["abg_tail"].append(torch.abs(abg_actual - abg_target).detach().cpu())
            if len(self.current_target["abg_tail"]) > self.steady_tail_len:
                self.current_target["abg_tail"].pop(0)

        self.num_steps += 1
        self.step_idx += 1

        if self.step_idx < self.total_steps:
            return None

        self._finalize_current_target()
        self.active = False
        base_env.commands_dog[:, 0:3] = 0.0

        denom = max(1, self.num_steps)
        mean_abs_dof_pos = (self.sum_abs_dof_pos_per_joint / denom).detach().cpu().tolist()
        ate_lin_x_pct = self.sum_pct_err_lin_x / self.num_valid_lin_x if self.num_valid_lin_x > 0 else 0.0
        ate_lin_y_pct = self.sum_pct_err_lin_y / self.num_valid_lin_y if self.num_valid_lin_y > 0 else 0.0
        ate_yaw_pct = self.sum_pct_err_yaw_z / self.num_valid_yaw_z if self.num_valid_yaw_z > 0 else 0.0

        result = {
            "steps": self.num_steps,
            "ate_lin_x": ate_lin_x_pct,
            "ate_lin_y": ate_lin_y_pct,
            "ate_ang_yaw": ate_yaw_pct,
            "tracking_dof_pos_per_joint": mean_abs_dof_pos,
            "num_joints": self.num_joints,
            "survival_rate": 1.0 if self.min_base_height > self.survival_height_thr else 0.0,
            "dog_only": self.dog_only,
        }

        if not self.dog_only:
            arm_denom = max(1, self.arm_samples)
            area_xy = _convex_hull_area_2d([(p[0], p[1]) for p in self.workspace_points])
            area_xz = _convex_hull_area_2d([(p[0], p[2]) for p in self.workspace_points])
            area_yz = _convex_hull_area_2d([(p[1], p[2]) for p in self.workspace_points])

            # Convergence curve from prefix target set
            conv_n, conv_xy, conv_xz, conv_yz, conv_solv = [], [], [], [], []
            solved_prefix = 0
            prefix_points = []
            for i, rec in enumerate(self.target_records, start=1):
                if rec["solved"]:
                    solved_prefix += 1
                    prefix_points.append(rec["xyz"])
                conv_n.append(i)
                conv_xy.append(_convex_hull_area_2d([(p[0], p[1]) for p in prefix_points]))
                conv_xz.append(_convex_hull_area_2d([(p[0], p[2]) for p in prefix_points]))
                conv_yz.append(_convex_hull_area_2d([(p[1], p[2]) for p in prefix_points]))
                conv_solv.append(solved_prefix / i)

            l_edges = [0.2, 0.35, 0.5, 0.65, 0.8]
            p_edges = [0.2, 0.35, 0.5, 0.65, 0.8]
            y_edges = [-0.6, -0.3, 0.0, 0.3, 0.6]
            l_records = [{"v": r["l"], "d": r["d"], "phi": r["phi"], "solved": r["solved"]} for r in self.target_records]
            p_records = [{"v": r["p"], "d": r["d"], "phi": r["phi"], "solved": r["solved"]} for r in self.target_records]
            y_records = [{"v": r["y"], "d": r["d"], "phi": r["phi"], "solved": r["solved"]} for r in self.target_records]

            result.update(
                {
                    "workspace_area_xy": area_xy,
                    "workspace_area_xz": area_xz,
                    "workspace_area_yz": area_yz,
                    "position_tracking_error_D": self.pos_err_sum / arm_denom,
                    "orientation_tracking_error_Phi": self.ori_err_sum / arm_denom,
                    "position_tracking_error_lpy": (self.lpy_err_sum / arm_denom).tolist(),
                    "orientation_tracking_error_abg": (self.rpy_err_sum / arm_denom).tolist(),
                    "solvability": self.solved_samples / arm_denom,
                    "conv_curve_n": conv_n,
                    "conv_curve_workspace_xy": conv_xy,
                    "conv_curve_workspace_xz": conv_xz,
                    "conv_curve_workspace_yz": conv_yz,
                    "conv_curve_solvability": conv_solv,
                    "strata_l": self._build_stratified_stats(l_edges, l_records),
                    "strata_p": self._build_stratified_stats(p_edges, p_records),
                    "strata_y": self._build_stratified_stats(y_edges, y_records),
                }
            )
        return result
