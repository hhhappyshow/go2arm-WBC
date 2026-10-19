from typing import Union

from params_proto import Meta

from go1_gym.envs.automatic.legged_robot_config import Cfg

def config_asset(Cnfg: Union[Cfg, Meta]):

    Cnfg.asset.file = '{MINI_GYM_ROOT_DIR}/resources/robots/arx5p2Go1/urdf/arx5p2Go1.urdf'
    
    # 设置末端执行器名称（用于动态查找 ee_idx）
    # arx5p2Go1 的末端执行器是 zarx_body8（最后一个机械臂 link，索引 31）
    Cnfg.asset.end_effector_name = "zarx_body6"
    
    Cnfg.asset.penalize_contacts_on = [
        'base', 'trunk',
        "arm", "wrist", 'zarx',
        "gripper", "thigh", "calf",
        "Head"
    ]
    
    Cnfg.asset.terminate_after_contacts_on = ['']
    
    Cnfg.asset.hip_joints = {'hip'}
    
    Cnfg.control.stiffness = {'joint': 35., 'widow': 5., "zarx": 5., "zarx_j3": 20}  # [N*m/rad]
    
    
    Cnfg.arm.control.stiffness_arm = {
            "zarx": 50,
            "zarx_j1": 40,
            "zarx_j2": 70,
            "zarx_j3": 70,
            "zarx_j4": 25,
            "zarx_j5": 25,
            "zarx_j6": 25,
            "zarx_j7": 50,
            "zarx_j8": 50,
    }  # [N*m/rad]
    Cnfg.arm.control.damping_arm = {
            "zarx": 20,
            "zarx_j1": 3,
            "zarx_j2": 15,
            "zarx_j3": 15,
            "zarx_j4": 2,
            "zarx_j5": 2,
            "zarx_j6": 2,
            "zarx_j7": 20,
            "zarx_j8": 20,
    }  # [N*m*s/rad]


    Cnfg.dog.control.stiffness_leg = {'joint': 35.}  # [N*m/rad]
    Cnfg.dog.control.damping_leg = {'joint': 1.}  # [N*m*s/rad]
   
    Cnfg.asset.render_sphere = True # NOTE no use in headless 

    Cnfg.init_state.default_joint_angles = {  # = target angles [rad] when action = 0.0
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
        'RR_calf_joint': -1.5,  # [rad]
        
        'widow_waist': 0., 
        'widow_shoulder': 0., 
        'widow_elbow': 0., 
        'forearm_roll': 0., 
        'widow_wrist_angle': 0., 
        'widow_wrist_rotate': 0., 
        'widow_forearm_roll': 0., 
        'gripper': 0., 
        'widow_left_finger': 0., 
        'widow_right_finger': 0.,
        
        "zarx_j1": 0.0,
        "zarx_j2": 0.8,
        "zarx_j3": 0.8,
        "zarx_j4": 0.0,
        "zarx_j5": 0.0,
        "zarx_j6": 0.0,
        "zarx_j7": 0.0,
        "zarx_j8": 0.0,
        
    }

def config_go2_airbot_asset(Cnfg: Union[Cfg, Meta]):
    """配置 go2_airbot 机器人（GO2 底盘 + Airbot 机械臂）"""
    
    Cnfg.asset.file = '{MINI_GYM_ROOT_DIR}/resources/robots/go2_airbot/urdf/go2_airbot.urdf'
    
    # 设置末端执行器名称（用于动态查找 ee_idx）
    Cnfg.asset.end_effector_name = "end_link"
    
    Cnfg.asset.penalize_contacts_on = [
        'base',
        "arm", "arm_link",
        "end_link",
        # "g2_", 
        "g2_base",
        "eef_connect",
        "gripper", "thigh", "calf",
        "Head"
    ]
    
    Cnfg.asset.terminate_after_contacts_on = ['']
    
    Cnfg.asset.hip_joints = {'hip'}
    
    Cnfg.control.stiffness = {'joint': 35., "arm": 5.}  # [N*m/rad]
    
    # Airbot 机械臂的刚度和阻尼配置
    # 包含所有可能的匹配键，确保 _process_dof_props 能够正确匹配
    Cnfg.arm.control.stiffness_arm = {
            # 通用匹配键
            "arm": 50,
            # 具体关节名称（go2_airbot）
            "arm_j1": 40,
            "arm_j2": 70,
            "arm_j3": 70,
            "arm_j4": 25,
            "arm_j5": 25,
            "arm_j6": 25,
    }  # [N*m/rad]
    Cnfg.arm.control.damping_arm = {
            # 通用匹配键
            "arm": 20,
            # 具体关节名称（go2_airbot）
            "arm_j1": 3,
            "arm_j2": 15,
            "arm_j3": 15,
            "arm_j4": 2,
            "arm_j5": 2,
            "arm_j6": 2,
    }  # [N*m*s/rad]

    Cnfg.dog.control.stiffness_leg = {'joint': 35.}  # [N*m/rad]
    Cnfg.dog.control.damping_leg = {'joint': 1.}  # [N*m*s/rad]
   
    Cnfg.asset.render_sphere = True # NOTE no use in headless 

    Cnfg.init_state.default_joint_angles = {  # = target angles [rad] when action = 0.0
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
        'RR_calf_joint': -1.5,  # [rad]
        
        # Airbot 机械臂部分
        "arm_j1": 0.0,
        "arm_j2": -0.3,
        "arm_j3": 0.3,
        "arm_j4": 0.0,
        "arm_j5": 0.0,
        "arm_j6": 0.0,
        # "g2_joint": 0.0,
        # "g2_left_joint": 0.0,
        # "g2_right_joint": 0.0,
        
    }