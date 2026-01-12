"""BXI ELF3 constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##
ELF3_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "bxi_elf3" / "xmls" / "elf3.xml"
)
assert ELF3_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, ELF3_XML.parent / "meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ELF3_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

# Motor specs (from Unitree).
ROTOR_INERTIAS_BXI50 = (
  (0.0760e-4 + 0.0300e-4),
  0.0892e-4,
  0.1807e-4,
)
GEARS_BXI50 = (
  1,
  1 + (65 / 19),
  1 + (65 / 19),
)
ARMATURE_BXI50 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_BXI50, GEARS_BXI50
)

ROTOR_INERTIAS_BXI70 = (
  (0.2620e-4 + 0.0802e-4),
  0.3099e-4,
  0.5617e-4,
)
GEARS_BXI70 = (
  1,
  1 + (65 / 19),
  1 + (65 / 19),
)
ARMATURE_BXI70 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_BXI70, GEARS_BXI70
)

ROTOR_INERTIAS_BXI85 = (
  (0.8590e-4 + 0.2683e-4),
  0.7604e-4,
  1.3499e-4,
)
GEARS_BXI85 = (
  1,
  1 + (65 / 19),
  1 + (65 / 19),
)
ARMATURE_BXI85 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_BXI85, GEARS_BXI85
)

ACTUATOR_BXI50 = ElectricActuator(
  reflected_inertia=ARMATURE_BXI50,
  velocity_limit=20.0,
  effort_limit=21.0,
)
ACTUATOR_BXI70 = ElectricActuator(
  reflected_inertia=ARMATURE_BXI70,
  velocity_limit=20.0,
  effort_limit=45.0,
)
ACTUATOR_BXI85 = ElectricActuator(
  reflected_inertia=ARMATURE_BXI85,
  velocity_limit=20.0,
  effort_limit=150.0,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_BXI50 = ARMATURE_BXI50 * NATURAL_FREQ**2
STIFFNESS_BXI70 = ARMATURE_BXI70 * NATURAL_FREQ**2
STIFFNESS_BXI85 = ARMATURE_BXI85 * NATURAL_FREQ**2

DAMPING_BXI50 = 2.0 * DAMPING_RATIO * ARMATURE_BXI50 * NATURAL_FREQ
DAMPING_BXI70 = 2.0 * DAMPING_RATIO * ARMATURE_BXI70 * NATURAL_FREQ
DAMPING_BXI85 = 2.0 * DAMPING_RATIO * ARMATURE_BXI85 * NATURAL_FREQ

ELF3_ACTUATOR_BXI50 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_z_joint",    
    ".*_wrist_x_joint",     
    ".*_wrist_y_joint",     
    ".*_wrist_z_joint", 
  ),
  stiffness=STIFFNESS_BXI50,
  damping=DAMPING_BXI50,
  effort_limit=ACTUATOR_BXI50.effort_limit,
  armature=ACTUATOR_BXI50.reflected_inertia,
)
ELF3_ACTUATOR_BXI70 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_z_joint", 
    ".*_shoulder_y_joint",
    ".*_shoulder_x_joint",
    ".*_elbow_y_joint",
  ),
  stiffness=STIFFNESS_BXI70,
  damping=DAMPING_BXI70,
  effort_limit=ACTUATOR_BXI70.effort_limit,
  armature=ACTUATOR_BXI70.reflected_inertia,
)
ELF3_ACTUATOR_BXI85 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "waist_z_joint",
    ".*_hip_y_joint", 
    ".*_hip_x_joint", 
    ".*_knee_y_joint"
  ),
  stiffness=STIFFNESS_BXI85,
  damping=DAMPING_BXI85,
  effort_limit=ACTUATOR_BXI85.effort_limit,
  armature=ACTUATOR_BXI85.reflected_inertia,
)

# Waist pitch/roll and ankles are 4-bar linkages with 2 BXI50 actuators.
# Due to the parallel linkage, the effective armature at the ankle and waist joints
# is configuration dependent. Since the exact geometry of the linkage is unknown, we
# assume a nominal 1:1 gear ratio. Under this assumption, the joint armature in the
# nominal configuration is approximated as the sum of the 2 actuators' armatures.
ELF3_ACTUATOR_WAIST_Y = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_y_joint",),
  stiffness=STIFFNESS_BXI70 * 2,
  damping=DAMPING_BXI70 * 2,
  effort_limit= 90,
  armature=ACTUATOR_BXI70.reflected_inertia * 2,
)
ELF3_ACTUATOR_WAIST_X = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_x_joint",),
  stiffness=STIFFNESS_BXI70 * 3,
  damping=DAMPING_BXI70 * 3,
  effort_limit= 100, # hardware clip # ACTUATOR_BXI70.effort_limit * 3,
  armature=ACTUATOR_BXI70.reflected_inertia * 3,
)
ELF3_ACTUATOR_ANKLE_Y = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_y_joint",),
  stiffness=STIFFNESS_BXI50 * 2,
  damping=DAMPING_BXI50 * 2,
  effort_limit=40,
  armature=ACTUATOR_BXI50.reflected_inertia * 2,
)
ELF3_ACTUATOR_ANKLE_X = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_x_joint",),
  stiffness= 10., #passive
  damping = 1.,
  effort_limit=15, #hardware clip #ACTUATOR_BXI50.effort_limit * 1.3,
  armature=ACTUATOR_BXI50.reflected_inertia * 1.3,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 1.1),
  joint_pos={
    ".*_hip_y_joint": -0.15,
    ".*_knee_y_joint": 0.3,
    ".*_ankle_y_joint": -0.15,
    ".*_shoulder_y_joint": 0.2,
    ".*_elbow_y_joint": 1.28,
    "l_shoulder_x_joint": 0.2,
    "r_shoulder_x_joint": -0.2,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 1.1),
  joint_pos={
    ".*_hip_y_joint": -0.3,
    ".*_knee_y_joint": 0.6,
    ".*_ankle_y_joint": 0.0,
    ".*_elbow_y_joint": 0.6,
    "l_shoulder_y_joint": 0.2,
    "l_shoulder_x_joint": 0.2,
    "r_shoulder_y_joint": 0.2,
    "r_shoulder_x_joint": -0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(l|r)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(l|r)_foot[1-7]_collision$": 1},
  friction={r"^(l|r)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(l|r)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(l|r)_foot[1-7]_collision$": 1},
  friction={r"^(l|r)_foot[1-7]_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(l|r)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

# joint_name_idx =[
#   "waist_y_joint" ,
#   "waist_x_joint" ,
#   "waist_z_joint" ,
#   "l_hip_y_joint" ,
#   "l_hip_x_joint" ,
#   "l_hip_z_joint" ,
#   "l_knee_y_joint" ,
#   "l_ankle_y_joint" ,
#   "l_ankle_x_joint" ,
#   "r_hip_y_joint" ,
#   "r_hip_x_joint" ,
#   "r_hip_z_joint" ,
#   "r_knee_y_joint" ,
#   "r_ankle_y_joint" ,
#   "r_ankle_x_joint" ,
#   "l_shoulder_y_joint",
#   "l_shoulder_x_joint",
#   "l_shoulder_z_joint",
#   "l_elbow_y_joint" ,
#   "l_wrist_x_joint" ,
#   "l_wrist_y_joint" ,
#   "l_wrist_z_joint" ,
#   "r_shoulder_y_joint",
#   "r_shoulder_x_joint",
#   "r_shoulder_z_joint",
#   "r_elbow_y_joint" ,
#   "r_wrist_x_joint" ,
#   "r_wrist_y_joint" ,
#   "r_wrist_z_joint",
# ]

##
# Final config.
##

# ELF3_ARTICULATION = EntityArticulationInfoCfg(
#   actuators=(
#     ELF3_ACTUATOR_BXI50,
#     ELF3_ACTUATOR_BXI70,
#     ELF3_ACTUATOR_BXI85,
#     ELF3_ACTUATOR_WAIST_Y,
#     ELF3_ACTUATOR_WAIST_X,
#     ELF3_ACTUATOR_ANKLE_Y,
#     ELF3_ACTUATOR_ANKLE_X,
#   ),
#   soft_joint_pos_limit_factor=0.9,
# )

# 为每个关节创建单独的配置
waist_y_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_y_joint",),
  stiffness=ELF3_ACTUATOR_WAIST_Y.stiffness,
  damping=ELF3_ACTUATOR_WAIST_Y.damping,
  effort_limit=ELF3_ACTUATOR_WAIST_Y.effort_limit,
  armature=ELF3_ACTUATOR_WAIST_Y.armature,
)

waist_x_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_x_joint",),
  stiffness=ELF3_ACTUATOR_WAIST_X.stiffness,
  damping=ELF3_ACTUATOR_WAIST_X.damping,
  effort_limit=ELF3_ACTUATOR_WAIST_X.effort_limit,
  armature=ELF3_ACTUATOR_WAIST_X.armature,
)

waist_z_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_z_joint",),
  stiffness=ELF3_ACTUATOR_BXI85.stiffness,
  damping=ELF3_ACTUATOR_BXI85.damping,
  effort_limit=ELF3_ACTUATOR_BXI85.effort_limit,
  armature=ELF3_ACTUATOR_BXI85.armature,
)

l_hip_y_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("l_hip_y_joint",),
  stiffness=ELF3_ACTUATOR_BXI85.stiffness,
  damping=ELF3_ACTUATOR_BXI85.damping,
  effort_limit=ELF3_ACTUATOR_BXI85.effort_limit,
  armature=ELF3_ACTUATOR_BXI85.armature,
)

l_hip_x_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("l_hip_x_joint",),
  stiffness=ELF3_ACTUATOR_BXI85.stiffness,
  damping=ELF3_ACTUATOR_BXI85.damping,
  effort_limit=ELF3_ACTUATOR_BXI85.effort_limit,
  armature=ELF3_ACTUATOR_BXI85.armature,
)

l_hip_z_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("l_hip_z_joint",),
  stiffness=ELF3_ACTUATOR_BXI70.stiffness,
  damping=ELF3_ACTUATOR_BXI70.damping,
  effort_limit=ELF3_ACTUATOR_BXI70.effort_limit,
  armature=ELF3_ACTUATOR_BXI70.armature,
)

l_knee_y_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("l_knee_y_joint",),
  stiffness=ELF3_ACTUATOR_BXI85.stiffness,
  damping=ELF3_ACTUATOR_BXI85.damping,
  effort_limit=ELF3_ACTUATOR_BXI85.effort_limit,
  armature=ELF3_ACTUATOR_BXI85.armature,
)

l_ankle_y_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("l_ankle_y_joint",),
  stiffness=ELF3_ACTUATOR_ANKLE_Y.stiffness,
  damping=ELF3_ACTUATOR_ANKLE_Y.damping,
  effort_limit=ELF3_ACTUATOR_ANKLE_Y.effort_limit,
  armature=ELF3_ACTUATOR_ANKLE_Y.armature,
)

l_ankle_x_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("l_ankle_x_joint",),
  stiffness=ELF3_ACTUATOR_ANKLE_X.stiffness,
  damping=ELF3_ACTUATOR_ANKLE_X.damping,
  effort_limit=ELF3_ACTUATOR_ANKLE_X.effort_limit,
  armature=ELF3_ACTUATOR_ANKLE_X.armature,
)

r_hip_y_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("r_hip_y_joint",),
  stiffness=ELF3_ACTUATOR_BXI85.stiffness,
  damping=ELF3_ACTUATOR_BXI85.damping,
  effort_limit=ELF3_ACTUATOR_BXI85.effort_limit,
  armature=ELF3_ACTUATOR_BXI85.armature,
)

r_hip_x_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("r_hip_x_joint",),
  stiffness=ELF3_ACTUATOR_BXI85.stiffness,
  damping=ELF3_ACTUATOR_BXI85.damping,
  effort_limit=ELF3_ACTUATOR_BXI85.effort_limit,
  armature=ELF3_ACTUATOR_BXI85.armature,
)

r_hip_z_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("r_hip_z_joint",),
  stiffness=ELF3_ACTUATOR_BXI70.stiffness,
  damping=ELF3_ACTUATOR_BXI70.damping,
  effort_limit=ELF3_ACTUATOR_BXI70.effort_limit,
  armature=ELF3_ACTUATOR_BXI70.armature,
)

r_knee_y_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("r_knee_y_joint",),
  stiffness=ELF3_ACTUATOR_BXI85.stiffness,
  damping=ELF3_ACTUATOR_BXI85.damping,
  effort_limit=ELF3_ACTUATOR_BXI85.effort_limit,
  armature=ELF3_ACTUATOR_BXI85.armature,
)

r_ankle_y_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("r_ankle_y_joint",),
  stiffness=ELF3_ACTUATOR_ANKLE_Y.stiffness,
  damping=ELF3_ACTUATOR_ANKLE_Y.damping,
  effort_limit=ELF3_ACTUATOR_ANKLE_Y.effort_limit,
  armature=ELF3_ACTUATOR_ANKLE_Y.armature,
)

r_ankle_x_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("r_ankle_x_joint",),
  stiffness=ELF3_ACTUATOR_ANKLE_X.stiffness,
  damping=ELF3_ACTUATOR_ANKLE_X.damping,
  effort_limit=ELF3_ACTUATOR_ANKLE_X.effort_limit,
  armature=ELF3_ACTUATOR_ANKLE_X.armature,
)

l_shoulder_y_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("l_shoulder_y_joint",),
  stiffness=ELF3_ACTUATOR_BXI70.stiffness,
  damping=ELF3_ACTUATOR_BXI70.damping,
  effort_limit=ELF3_ACTUATOR_BXI70.effort_limit,
  armature=ELF3_ACTUATOR_BXI70.armature,
)

l_shoulder_x_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("l_shoulder_x_joint",),
  stiffness=ELF3_ACTUATOR_BXI70.stiffness,
  damping=ELF3_ACTUATOR_BXI70.damping,
  effort_limit=ELF3_ACTUATOR_BXI70.effort_limit,
  armature=ELF3_ACTUATOR_BXI70.armature,
)

l_shoulder_z_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("l_shoulder_z_joint",),
  stiffness=ELF3_ACTUATOR_BXI50.stiffness,
  damping=ELF3_ACTUATOR_BXI50.damping,
  effort_limit=ELF3_ACTUATOR_BXI50.effort_limit,
  armature=ELF3_ACTUATOR_BXI50.armature,
)

l_elbow_y_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("l_elbow_y_joint",),
  stiffness=ELF3_ACTUATOR_BXI70.stiffness,
  damping=ELF3_ACTUATOR_BXI70.damping,
  effort_limit=ELF3_ACTUATOR_BXI70.effort_limit,
  armature=ELF3_ACTUATOR_BXI70.armature,
)

l_wrist_x_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("l_wrist_x_joint",),
  stiffness=ELF3_ACTUATOR_BXI50.stiffness,
  damping=ELF3_ACTUATOR_BXI50.damping,
  effort_limit=ELF3_ACTUATOR_BXI50.effort_limit,
  armature=ELF3_ACTUATOR_BXI50.armature,
)

l_wrist_y_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("l_wrist_y_joint",),
  stiffness=ELF3_ACTUATOR_BXI50.stiffness,
  damping=ELF3_ACTUATOR_BXI50.damping,
  effort_limit=ELF3_ACTUATOR_BXI50.effort_limit,
  armature=ELF3_ACTUATOR_BXI50.armature,
)

l_wrist_z_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("l_wrist_z_joint",),
  stiffness=ELF3_ACTUATOR_BXI50.stiffness,
  damping=ELF3_ACTUATOR_BXI50.damping,
  effort_limit=ELF3_ACTUATOR_BXI50.effort_limit,
  armature=ELF3_ACTUATOR_BXI50.armature,
)

r_shoulder_y_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("r_shoulder_y_joint",),
  stiffness=ELF3_ACTUATOR_BXI70.stiffness,
  damping=ELF3_ACTUATOR_BXI70.damping,
  effort_limit=ELF3_ACTUATOR_BXI70.effort_limit,
  armature=ELF3_ACTUATOR_BXI70.armature,
)

r_shoulder_x_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("r_shoulder_x_joint",),
  stiffness=ELF3_ACTUATOR_BXI70.stiffness,
  damping=ELF3_ACTUATOR_BXI70.damping,
  effort_limit=ELF3_ACTUATOR_BXI70.effort_limit,
  armature=ELF3_ACTUATOR_BXI70.armature,
)

r_shoulder_z_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("r_shoulder_z_joint",),
  stiffness=ELF3_ACTUATOR_BXI50.stiffness,
  damping=ELF3_ACTUATOR_BXI50.damping,
  effort_limit=ELF3_ACTUATOR_BXI50.effort_limit,
  armature=ELF3_ACTUATOR_BXI50.armature,
)

r_elbow_y_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("r_elbow_y_joint",),
  stiffness=ELF3_ACTUATOR_BXI70.stiffness,
  damping=ELF3_ACTUATOR_BXI70.damping,
  effort_limit=ELF3_ACTUATOR_BXI70.effort_limit,
  armature=ELF3_ACTUATOR_BXI70.armature,
)

r_wrist_x_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("r_wrist_x_joint",),
  stiffness=ELF3_ACTUATOR_BXI50.stiffness,
  damping=ELF3_ACTUATOR_BXI50.damping,
  effort_limit=ELF3_ACTUATOR_BXI50.effort_limit,
  armature=ELF3_ACTUATOR_BXI50.armature,
)

r_wrist_y_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("r_wrist_y_joint",),
  stiffness=ELF3_ACTUATOR_BXI50.stiffness,
  damping=ELF3_ACTUATOR_BXI50.damping,
  effort_limit=ELF3_ACTUATOR_BXI50.effort_limit,
  armature=ELF3_ACTUATOR_BXI50.armature,
)

r_wrist_z_joint = BuiltinPositionActuatorCfg(
  target_names_expr=("r_wrist_z_joint",),
  stiffness=ELF3_ACTUATOR_BXI50.stiffness,
  damping=ELF3_ACTUATOR_BXI50.damping,
  effort_limit=ELF3_ACTUATOR_BXI50.effort_limit,
  armature=ELF3_ACTUATOR_BXI50.armature,
)

# 新的关节配置表，按joint_name_idx顺序排列
ELF3_ARTICULATION_NEW = EntityArticulationInfoCfg(
  actuators=(
    waist_y_joint,
    waist_x_joint,
    waist_z_joint,
    l_hip_y_joint,
    l_hip_x_joint,
    l_hip_z_joint,
    l_knee_y_joint,
    l_ankle_y_joint,
    l_ankle_x_joint,
    r_hip_y_joint,
    r_hip_x_joint,
    r_hip_z_joint,
    r_knee_y_joint,
    r_ankle_y_joint,
    r_ankle_x_joint,
    l_shoulder_y_joint,
    l_shoulder_x_joint,
    l_shoulder_z_joint,
    l_elbow_y_joint,
    l_wrist_x_joint,
    l_wrist_y_joint,
    l_wrist_z_joint,
    r_shoulder_y_joint,
    r_shoulder_x_joint,
    r_shoulder_z_joint,
    r_elbow_y_joint,
    r_wrist_x_joint,
    r_wrist_y_joint,
    r_wrist_z_joint,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_elf3_robot_cfg() -> EntityCfg:
  """Get a fresh ELF3 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=ELF3_ARTICULATION_NEW,
  )


ELF3_ACTION_SCALE: dict[str, float] = {}
for a in ELF3_ARTICULATION_NEW.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    ELF3_ACTION_SCALE[n] = 0.25 * e / s
    # if (n == "l_ankle_x_joint") or (n == "r_ankle_x_joint"):
    #   ELF3_ACTION_SCALE[n] /= 2


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_elf3_robot_cfg())

  path = Path("./xmls/elf3_complie.xml")

  robot.write_xml(path)

  viewer.launch(robot.spec.compile())
