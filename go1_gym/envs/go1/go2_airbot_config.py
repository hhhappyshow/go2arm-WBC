"""
go2_airbot 机器人专用配置文件
解决关节名称映射和配置问题
"""
from typing import Union
from params_proto import Meta
from go1_gym.envs.automatic.legged_robot_config import Cfg


def config_go2_airbot_complete(Cnfg: Union[Cfg, Meta]):
    """
    完整的 go2_airbot 配置函数
    确保所有必要的配置项都正确设置，包括关节名称映射
    """
    
    # 设置 URDF 文件路径
    Cnfg.asset.file = '{MINI_GYM_ROOT_DIR}/resources/robots/go2_airbot/urdf/go2_airbot.urdf'
    
    # 碰撞惩罚配置
    Cnfg.asset.penalize_contacts_on = [
        'base',
        "arm", "arm_link",
        "g2_", "eef_connect",
        "gripper", "thigh", "calf",
        "Head"
    ]
    
    Cnfg.asset.terminate_after_contacts_on = ['']
    Cnfg.asset.hip_joints = {'hip'}
    
    # 控制刚度配置 - 使用通用匹配模式
    Cnfg.control.stiffness = {
        'joint': 35., 
        "arm": 5., 
        "arm_j1": 40,
        "arm_j2": 50,
        "arm_j3": 50,
        "arm_j4": 30,
        "arm_j5": 30,
        "arm_j6": 30,
    }  # [N*m/rad]
    
    # 机械臂刚度配置 - 包含所有可能的关节名称模式
    Cnfg.arm.control.stiffness_arm = {
        # 通用匹配键
        "arm": 50,
        # 具体关节名称（go2_airbot）
        "arm_j1": 40,
        "arm_j2": 50,
        "arm_j3": 50,
        "arm_j4": 30,
        "arm_j5": 30,
        "arm_j6": 30,
    }  # [N*m/rad]
    
    # 机械臂阻尼配置
    Cnfg.arm.control.damping_arm = {
        # 通用匹配键
        "arm": 20,
        # 具体关节名称（go2_airbot）
        "arm_j1": 3,
        "arm_j2": 10,
        "arm_j3": 10,
        "arm_j4": 5,
        "arm_j5": 5,
        "arm_j6": 5,
    }  # [N*m*s/rad]

    # 四足部分配置
    Cnfg.dog.control.stiffness_leg = {'joint': 35.}  # [N*m/rad]
    Cnfg.dog.control.damping_leg = {'joint': 1.}  # [N*m*s/rad]
   
    Cnfg.asset.render_sphere = True  # NOTE no use in headless 

    # 初始关节角度配置
    Cnfg.init_state.default_joint_angles = {
        # GO2 四足部分
        'FL_hip_joint': 0.1,  # [rad]
        'RL_hip_joint': 0.1,  # [rad]
        'FR_hip_joint': -0.1,  # [rad]
        'RR_hip_joint': -0.1,  # [rad]

        'FL_thigh_joint': 0.8,  # [rad]
        'RL_thigh_joint': 1.,  # [rad]
        'FR_thigh_joint': 0.8,  # [rad]
        'RR_thigh_joint': 1.,  # [rad]

        'FL_calf_joint': -1.5,  # [rad]
        'RL_calf_joint': -1.5,  # [rad]
        'FR_calf_joint': -1.5,  # [rad]
        'RR_calf_joint': -2.2,  # [rad]
        
        # Airbot 机械臂部分
        "arm_j1": 0.0,
        "arm_j2": 0.8,
        "arm_j3": 0.8,
        "arm_j4": 0.0,
        "arm_j5": 0.0,
        "arm_j6": 0.0,
        "g2_joint": 0.0,
        "g2_left_joint": 0.0,
        "g2_right_joint": 0.0,
    }


def get_joint_config_mapping(robot_type: str = "go2_airbot"):
    """
    获取关节名称到配置键的映射关系
    用于在 _process_dof_props 中动态匹配关节名称
    """
    if robot_type == "go2_airbot":
        return {
            # 实际关节名称 -> 配置键
            "arm_j1": "arm_j1",
            "arm_j2": "arm_j2",
            "arm_j3": "arm_j3",
            "arm_j4": "arm_j4",
            "arm_j5": "arm_j5",
            "arm_j6": "arm_j6",
        }
    else:
        # 默认映射（go2_airbot）
        return {
            f"arm_j{i}": f"arm_j{i}" for i in range(1, 7)
        }

