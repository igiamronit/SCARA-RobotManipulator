"""Structural checks only: joint directions, DH tool position, and model loading."""

import json
import math
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
P = json.loads((ROOT / "config" / "robot.json").read_text(encoding="utf-8"))
m = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "scara.xml"))
u = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "scara.urdf"))
assert m.nq == 6 and m.nu == 6
assert u.nq == 6
data = mujoco.MjData(m)
names = ["shoulder", "elbow", "vertical", "wrist_roll", "left_grip", "right_grip"]
assert [m.joint(i).name for i in range(6)] == names
assert [m.joint(i).type for i in range(6)] == [3,3,2,3,2,2]


def pose(q):
    data.qpos[:] = q
    mujoco.mj_forward(m, data)
    return data.site("tool").xpos.copy(), data.site("tool").xmat.reshape(3,3).copy()


def dh_position(q):
    q1,q2,q3,q4,_,_ = q
    return np.array([
        P["a1_m"]*math.cos(q1)+P["a2_m"]*math.cos(q1+q2),
        P["a1_m"]*math.sin(q1)+P["a2_m"]*math.sin(q1+q2),
        P["base_height_m"]-P["d3_min_m"]-q3-P["d4_m"],
    ])


samples = [
    [0,0,0,0,0,0],
    [math.radians(45),math.radians(-55),.04,math.radians(30),.01,.01],
    [math.radians(-90),math.radians(70),.11,math.radians(-110),0,.018],
]
for q in samples:
    actual, orientation = pose(q)
    assert np.linalg.norm(actual-dh_position(q)) < 2e-7, (actual,dh_position(q))
    assert np.linalg.det(orientation) > .999999

zero = np.zeros(6)
tool0,_ = pose(zero)
vertical = zero.copy(); vertical[2]=.01
tool1,_ = pose(vertical)
assert np.allclose(tool1-tool0,[0,0,-.01],atol=1e-9)
wrist=zero.copy(); wrist[3]=.4
tool2,_=pose(wrist)
assert np.allclose(tool2,tool0,atol=1e-9)
for idx in (4,5):
    q=zero.copy(); q[idx]=.01
    pose(q)
    finger_name="left_finger" if idx==4 else "right_finger"
    finger=data.body(finger_name).xpos.copy()
    pose(zero)
    initial=data.body(finger_name).xpos.copy()
    assert abs(abs(finger[1]-initial[1])-.01)<1e-9

# Separate physical check: the finger motors actually open both fingers.
motion = mujoco.MjData(m)
motion.qpos[2] = .05
motion.ctrl[4:6] = P["effort_limit"]["each_finger_N"]
for _ in range(100):
    mujoco.mj_step(m, motion)
assert .017 < motion.qpos[4] < .019
assert .017 < motion.qpos[5] < .019

print("PASS: MJCF and URDF load; six joint types/order correct; DH tool positions and joint directions agree.")
print("PASS: both gripper motors open the fingers in a 100 ms physics step check.")
print("This does not validate object grasp, arm trajectory tracking, or cycle throughput.")
