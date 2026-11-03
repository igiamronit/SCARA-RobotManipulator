"""One MuJoCo window for FK, IK, and forward velocity kinematics.

This viewer poses joints using mj_forward. It never steps physics or claims
actuator tracking. The project's equations remain in scara_kinematics.py.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import sys
import time

import glfw
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "kinematics"))
from scara_kinematics import (forward_kinematics, inverse_kinematics,
                              task_jacobian, UnreachableTargetError, JointLimitError)

HOME = np.array([math.radians(25), math.radians(-65), .045, math.radians(20)])
FACTORS = np.array([180 / math.pi, 180 / math.pi, 1000, 180 / math.pi])
MODES = ("Joints", "Target", "Velocity")


class State:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "scara.xml"))
        self.data = mujoco.MjData(self.model)
        self.raw = json.loads((ROOT / "config" / "robot.json").read_text())
        self.limits = self.model.jnt_range[:4].copy()
        rates = self.raw["velocity_limit"]
        self.max_rates = np.minimum([rates["shoulder_rad_s"], rates["elbow_rad_s"],
                                     rates["vertical_m_s"], rates["wrist_rad_s"]],
                                    [math.radians(60), math.radians(60), .06, math.radians(90)])
        self.mode = 0
        self.axes = True
        self.trail_visible = True
        self.grip_open = True
        self.branch = -1
        self.running = False
        self.circle = False
        self.phase = 0.0
        self.rates = np.zeros(4)
        self.actual_rates = np.zeros(4)
        self.trail = deque(maxlen=240)
        self.home()

    @property
    def q(self):
        return self.data.qpos[:4].copy()

    @property
    def pose(self):
        return forward_kinematics(self.q)

    @property
    def velocity(self):
        return task_jacobian(self.q) @ self.actual_rates

    def pose_joints(self, q):
        self.data.qpos[:4] = np.clip(q, self.limits[:, 0], self.limits[:, 1])
        self.data.qpos[4:6] = self.raw["finger_travel_m"] if self.grip_open else 0
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        point = self.data.site("tool").xpos.copy()
        if not self.trail or np.linalg.norm(point - self.trail[-1]) > .001:
            self.trail.append(point)

    def home(self):
        self.rates[:] = 0
        self.actual_rates[:] = 0
        self.running = self.circle = False
        self.branch = -1
        self.trail.clear()
        self.pose_joints(HOME)
        self.target = self.pose.copy()
        self.valid = True
        self.status = "Ready"

    def set_mode(self, index):
        self.mode = index
        self.running = self.circle = False
        self.rates[:] = 0
        self.actual_rates[:] = 0
        self.target = self.pose.copy()
        self.branch = 1 if self.q[1] >= 0 else -1
        self.valid = True
        self.status = "Ready"
        self.trail.clear()

    def set_target(self):
        self.actual_rates[:] = 0
        try:
            solutions = inverse_kinematics(self.target)
            preferred = [q for q in solutions if q[1] * self.branch >= -1e-9]
            if not preferred:
                self.valid = False
                self.status = "This elbow branch hits a joint limit"
                return
            self.pose_joints(min(preferred, key=lambda q: np.linalg.norm(q - self.q)))
            self.valid = True
            self.status = "Target reached"
        except UnreachableTargetError:
            self.valid = False
            self.status = "Target outside geometric workspace"
        except JointLimitError:
            self.valid = False
            self.status = "Target excluded by joint limits"

    def set_control(self, index, value):
        self.actual_rates[:] = 0
        if self.mode == 0:
            q = self.q
            q[index] = value
            self.pose_joints(q)
            self.target = self.pose.copy()
            self.status = "Joint position updated"
        elif self.mode == 1:
            self.circle = False
            self.target[index] = value
            self.set_target()
        else:
            self.rates[index] = np.clip(value, -self.max_rates[index], self.max_rates[index])

    def tick(self, dt):
        self.actual_rates[:] = 0
        if self.mode == 2 and self.running and dt > 0:
            before = self.q
            after = np.clip(before + dt * self.rates, self.limits[:, 0], self.limits[:, 1])
            self.pose_joints(after)
            self.actual_rates = (after - before) / dt
            clipped = not np.allclose(after, before + dt * self.rates, atol=1e-12, rtol=0)
            self.status = "Joint limit: motion clipped" if clipped else "Joint velocities running"
        elif self.mode == 1 and self.circle:
            self.phase += .55 * dt
            self.target[:] = [.29 + .035 * math.cos(self.phase),
                              .035 * math.sin(self.phase), .08, 0]
            self.set_target()

    def controls(self):
        if self.mode == 0:
            return self.q * FACTORS, self.limits * FACTORS[:, None], ["Shoulder (deg)", "Elbow (deg)", "Slide (mm)", "Wrist (deg)"]
        if self.mode == 1:
            zmax = self.raw["base_height_m"] - self.raw["d3_min_m"] - self.raw["d4_m"]
            zmin = self.raw["base_height_m"] - self.raw["d3_max_m"] - self.raw["d4_m"]
            reach = self.raw["a1_m"] + self.raw["a2_m"]
            limits = np.array([[-reach-.03, reach+.03], [-reach-.03, reach+.03],
                               [zmin-.015, zmax+.015], [-180, 180]])
            limits[:3] *= 1000
            values = self.target * [1000, 1000, 1000, 180/math.pi]
            return values, limits, ["Target X (mm)", "Target Y (mm)", "Target Z (mm)", "Target yaw (deg)"]
        bounds = self.max_rates * FACTORS
        return self.rates * FACTORS, np.c_[-bounds, bounds], ["Shoulder (deg/s)", "Elbow (deg/s)", "Slide (mm/s)", "Wrist (deg/s)"]

    def edit_display(self, index, value):
        factor = ([1000, 1000, 1000, 180/math.pi] if self.mode == 1 else FACTORS)[index]
        self.set_control(index, value / factor)


class App:
    def __init__(self, state, hidden=False):
        if not glfw.init():
            raise RuntimeError("GLFW could not open an OpenGL display.")
        glfw.window_hint(glfw.VISIBLE, not hidden)
        glfw.window_hint(glfw.SAMPLES, 4)
        self.window = glfw.create_window(1380, 850, "SCARA - Kinematics", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create the MuJoCo window.")
        glfw.set_window_size_limits(self.window, 1050, 780, glfw.DONT_CARE, glfw.DONT_CARE)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.state = state
        self.scene = mujoco.MjvScene(state.model, maxgeom=1800)
        self.context = mujoco.MjrContext(state.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [.17, 0, .17]
        self.camera.distance = .85
        self.camera.azimuth = 115
        self.camera.elevation = -25
        self.options = mujoco.MjvOption()
        self.options.sitegroup[:] = 0
        self.buttons = []
        self.sliders = []
        self.dragging = None
        self.rotating = False
        self.last_cursor = (0, 0)
        self.selected = 0
        self.width, self.height = 1380, 850
        self.scale = 1
        glfw.set_mouse_button_callback(self.window, self.mouse)
        glfw.set_cursor_pos_callback(self.window, self.cursor)
        glfw.set_scroll_callback(self.window, self.scroll)
        glfw.set_key_callback(self.window, self.key)
        glfw.set_window_focus_callback(self.window, self.focus)

    def focus(self, window, focused):
        if not focused:
            if self.state.running or self.state.circle:
                self.state.status = "Paused: window lost focus"
            self.state.running = False
            self.state.circle = False
            self.state.actual_rates[:] = 0
            self.dragging = None
            self.rotating = False

    def rect(self, x, y, w, h):
        s = self.scale
        return mujoco.MjrRect(int(x*s), int(self.height-(y+h)*s), int(w*s), int(h*s))

    def fill(self, x, y, w, h, color):
        mujoco.mjr_rectangle(self.rect(x,y,w,h), *color, 1)

    def text(self, x, y, value, w=334, h=23):
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           self.rect(x,y,w,h), value, "", self.context)

    def button(self, x, y, w, label, action, active=False):
        color = (.30,.36,.40) if active else (.22,.23,.25)
        mujoco.mjr_label(self.rect(x,y,w,30), mujoco.mjtFont.mjFONT_NORMAL,
                        label, *color, 1, .95,.95,.95, self.context)
        self.buttons.append((x,y,w,30,action))

    def slider(self, index, y, value, low, high, label):
        self.text(12,y,f"{label}   {value: .1f}")
        self.fill(23,y+33,302,3,(.42,.43,.45))
        fraction = float(np.clip((value-low)/(high-low),0,1))
        self.fill(23+302*fraction-5,y+27,10,15,(.82,.83,.85))
        self.text(13,y+43,f"{low:.0f}",w=120,h=18)
        self.text(265,y+43,f"{high:.0f}",w=72,h=18)
        if self.selected == index:
            self.fill(8,y+6,2,15,(.8,.8,.8))
        self.sliders.append((index,23,y+23,302,25,low,high))

    def panel(self):
        s = self.state
        self.fill(0,0,350,850,(.12,.13,.14))
        self.text(12,12,"SCARA / Kinematics")
        self.text(12,38,"Four arm joints | SI model")
        self.buttons.clear()
        self.sliders.clear()
        for i,label in enumerate(MODES):
            self.button(13+109*i,76,103,label,lambda i=i:s.set_mode(i),s.mode==i)
        descriptions = ["Joint positions -> tool pose", "Tool target -> joint positions", "Joint speeds -> tool velocity"]
        self.text(12,115,descriptions[s.mode])
        values,limits,labels = s.controls()
        for i in range(4):
            self.slider(i,150+65*i,values[i],*limits[i],labels[i])
        if s.mode == 0:
            self.button(13,420,158,"Straight arm",self.straight)
            self.button(179,420,158,"Reset joints",s.home)
        elif s.mode == 1:
            branch = "Elbow +" if s.branch > 0 else "Elbow -"
            self.button(13,420,158,branch+" / switch",self.branch)
            self.button(179,420,158,"Stop circle" if s.circle else "Trace circle",self.circle,s.circle)
        else:
            self.button(13,420,158,"Pause" if s.running else "Run velocities",self.run,s.running)
            self.button(179,420,158,"Zero velocities",self.zero)
        self.button(13,461,158,"Home",s.home)
        self.button(179,461,158,"Close gripper" if s.grip_open else "Open gripper",self.grip)
        self.button(13,502,158,"Axes: on" if s.axes else "Axes: off",self.axes,s.axes)
        self.button(179,502,158,"Trail: on" if s.trail_visible else "Trail: off",self.trail,s.trail_visible)
        self.fill(13,546,324,1,(.35,.36,.37))
        p=s.pose; v=s.velocity
        self.text(12,558,"Tool position (world)")
        self.text(12,583,f"X {p[0]*1000:6.1f}  Y {p[1]*1000:6.1f}  Z {p[2]*1000:6.1f} mm")
        self.text(12,608,f"Yaw {math.degrees(p[3]):.1f} deg")
        if s.mode == 2:
            self.text(12,641,f"Vx {v[0]*1000:6.1f}  Vy {v[1]*1000:6.1f} mm/s")
            self.text(12,666,f"Vz {v[2]*1000:6.1f} mm/s  yaw {math.degrees(v[3]):.1f} deg/s")
        elif s.mode == 1:
            q = s.q * FACTORS
            self.text(12,641,f"q1 {q[0]:.1f}  q2 {q[1]:.1f} deg")
            self.text(12,666,f"q3 {q[2]:.1f} mm  q4 {q[3]:.1f} deg")
        else:
            self.text(12,641,"Open Velocity tab to inspect J qdot")
        measure=abs(math.sin(s.q[1]))
        singular = "SINGULAR" if measure<1e-8 else "Near singular" if measure<.12 else "Regular"
        self.text(12,704,f"{singular} | abs(sin q2) = {measure:.3f}")
        self.text(12,737,s.status)
        self.text(12,778,"F1/F2/F3: mode   1-4: select")
        self.text(12,800,"Left/Right: adjust   H: home")
        self.text(12,822,"Mouse drag: camera   Wheel: zoom")

    def straight(self):
        self.state.pose_joints([0,0,.045,0])
        self.state.actual_rates[:] = 0
        self.state.status = "Straight arm: singular configuration"

    def branch(self):
        self.state.branch *= -1
        self.state.circle = False
        self.state.trail.clear()
        self.state.set_target()

    def circle(self):
        self.state.circle = not self.state.circle
        if self.state.circle:
            self.state.phase = 0
            self.state.trail.clear()

    def run(self):
        self.state.running = not self.state.running
        if not self.state.running:
            self.state.actual_rates[:] = 0
            self.state.status = "Paused"

    def zero(self):
        self.state.rates[:] = 0
        self.state.actual_rates[:] = 0
        self.state.running = False
        self.state.status = "Velocities zeroed"

    def grip(self):
        self.state.grip_open = not self.state.grip_open
        self.state.pose_joints(self.state.q)

    def axes(self):
        self.state.axes = not self.state.axes

    def trail(self):
        self.state.trail_visible = not self.state.trail_visible
        self.state.trail.clear()

    def point(self):
        x,y=glfw.get_cursor_pos(self.window)
        ww,wh=glfw.get_window_size(self.window)
        return x*self.width/ww/self.scale,y*self.height/wh/self.scale

    def mouse(self,window,button,action,mods):
        if action==glfw.RELEASE:
            self.dragging=None
            self.rotating=False
            return
        x,y=self.point()
        if x>=350:
            self.rotating=True
            self.last_cursor=(x,y)
            return
        if button!=glfw.MOUSE_BUTTON_LEFT:
            return
        for bx,by,bw,bh,callback in self.buttons:
            if bx<=x<=bx+bw and by<=y<=by+bh:
                callback()
                return
        for slider in self.sliders:
            i,sx,sy,sw,sh,low,high=slider
            if sx-8<=x<=sx+sw+8 and sy<=y<=sy+sh:
                self.dragging=slider
                self.selected=i
                self.adjust_slider(x)
                return

    def adjust_slider(self,x):
        i,sx,sy,sw,sh,low,high=self.dragging
        self.state.edit_display(i,low+np.clip((x-sx)/sw,0,1)*(high-low))

    def cursor(self,window,x,y):
        x,y=self.point()
        if self.dragging is not None:
            self.adjust_slider(x)
        elif self.rotating:
            dx,dy=x-self.last_cursor[0],y-self.last_cursor[1]
            shift=glfw.get_key(window,glfw.KEY_LEFT_SHIFT)==glfw.PRESS
            action=mujoco.mjtMouse.mjMOUSE_MOVE_V if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            mujoco.mjv_moveCamera(self.state.model,action,dx/850,dy/850,self.camera)
            self.last_cursor=(x,y)

    def scroll(self,window,x,y):
        if self.point()[0]>=350:
            mujoco.mjv_moveCamera(self.state.model,mujoco.mjtMouse.mjMOUSE_ZOOM,0,-.05*y,self.camera)

    def key(self,window,key,scancode,action,mods):
        if action not in (glfw.PRESS,glfw.REPEAT):
            return
        if key==glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window,True)
        elif glfw.KEY_F1<=key<=glfw.KEY_F3:
            self.state.set_mode(key-glfw.KEY_F1)
        elif glfw.KEY_1<=key<=glfw.KEY_4:
            self.selected=key-glfw.KEY_1
        elif key==glfw.KEY_H:
            self.state.home()
        elif key==glfw.KEY_SPACE:
            if self.state.mode==2:
                self.run()
            elif self.state.mode==1:
                self.circle()
        elif key==glfw.KEY_B and self.state.mode==1:
            self.branch()
        elif key in (glfw.KEY_LEFT,glfw.KEY_RIGHT):
            values,limits,_=self.state.controls()
            i=self.selected
            step=(limits[i,1]-limits[i,0])/100
            sign=1 if key==glfw.KEY_RIGHT else -1
            self.state.edit_display(i,float(np.clip(values[i]+sign*step,*limits[i])))

    def extra_geom(self,kind,pos,size,color,end=None):
        if self.scene.ngeom>=self.scene.maxgeom:
            return
        geom=self.scene.geoms[self.scene.ngeom]
        mujoco.mjv_initGeom(geom,kind,np.asarray(size,dtype=float),np.asarray(pos,dtype=float),
                           np.eye(3).ravel(),np.asarray(color,dtype=np.float32))
        if end is not None:
            mujoco.mjv_connector(geom,kind,size[0],np.asarray(pos,dtype=float),np.asarray(end,dtype=float))
        self.scene.ngeom+=1

    def render(self):
        self.width,self.height=glfw.get_framebuffer_size(self.window)
        if self.width==0 or self.height==0:
            return
        self.scale=self.height/850
        sidebar=int(350*self.scale)
        viewport=mujoco.MjrRect(sidebar,0,self.width-sidebar,self.height)
        mujoco.mjv_updateScene(self.state.model,self.state.data,self.options,None,self.camera,
                              mujoco.mjtCatBit.mjCAT_ALL,self.scene)
        s=self.state
        p=s.data.site("tool").xpos.copy()
        if s.trail_visible:
            points=list(s.trail)
            for a,b in zip(points,points[1:]):
                self.extra_geom(mujoco.mjtGeom.mjGEOM_LINE,a,[2,0,0],[.25,.65,.70,1],b)
        if s.axes:
            for axis,color in zip(np.eye(3),[(.9,.2,.2,1),(.2,.8,.3,1),(.2,.4,1,1)]):
                self.extra_geom(mujoco.mjtGeom.mjGEOM_ARROW,p,[.002,0,0],color,p+.045*axis)
        if s.mode==1:
            color=[.15,.85,.3,.8] if s.valid else [1,.15,.1,.9]
            target=s.target[:3]
            self.extra_geom(mujoco.mjtGeom.mjGEOM_SPHERE,target,[.008,0,0],color)
            end=target+.055*np.array([math.cos(s.target[3]),math.sin(s.target[3]),0])
            self.extra_geom(mujoco.mjtGeom.mjGEOM_ARROW,target,[.0025,0,0],color,end)
        if s.mode==2 and np.linalg.norm(s.velocity[:3])>1e-6:
            self.extra_geom(mujoco.mjtGeom.mjGEOM_ARROW,p,[.003,0,0],[1,.75,.15,1],p+.7*s.velocity[:3])
        mujoco.mjr_render(viewport,self.scene,self.context)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           viewport,"Kinematic visualization", "",self.context)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                           viewport,"Direct joint posing | no physics stepping", "",self.context)
        self.panel()

    def snapshot(self,path):
        from PIL import Image
        self.render()
        rgb=np.empty((self.height,self.width,3),dtype=np.uint8)
        mujoco.mjr_readPixels(rgb,None,mujoco.MjrRect(0,0,self.width,self.height),self.context)
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(np.flipud(rgb)).save(path)

    def close(self):
        self.context.free()
        glfw.destroy_window(self.window)
        glfw.terminate()


def check():
    s=State()
    s.set_control(0,.4)
    assert np.allclose(s.data.site("tool").xpos,s.pose[:3],atol=1e-12)
    s.set_mode(1)
    s.target=np.array([.30,.02,.075,.1])
    s.set_target()
    assert s.valid and np.allclose(s.pose,s.target,atol=1e-10)
    first=s.q
    s.branch*=-1
    s.set_target()
    assert s.valid and first[1]*s.q[1]<0
    previous=s.q
    s.target[0]=.6
    s.set_target()
    assert not s.valid and np.array_equal(previous,s.q)
    s.home()
    s.set_mode(2)
    s.rates[:]=[.2,-.15,.012,.1]
    s.running=True
    before=s.pose
    s.tick(1e-5)
    numeric=(s.pose-before)/1e-5
    assert np.allclose(numeric,s.velocity,atol=1e-6)
    s.pose_joints(s.limits[:,1])
    s.rates[:]=s.max_rates
    s.tick(.01)
    assert np.allclose(s.actual_rates,0)
    assert np.all(s.q<=s.limits[:,1])
    s.set_mode(1)
    assert not s.running and np.allclose(s.rates,0)
    s.home()
    s.circle=True
    s.set_mode(1)
    s.circle=True
    for _ in range(400):
        s.tick(.04)
        assert s.valid
        assert np.allclose(s.pose,s.target,atol=1e-10)
    print("PASS: FK edits, both IK branches, invalid target hold, velocity mapping, limits, and circle.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true",help="Check state transitions without a window")
    parser.add_argument("--snapshot",type=Path,help="Save one hidden-window screenshot and exit")
    parser.add_argument("--mode",choices=["joints","target","velocity"],default="joints")
    parser.add_argument("--duration",type=float,default=0,help="Optional auto-close time in seconds")
    args=parser.parse_args()
    if args.check:
        check()
        return
    state=State()
    state.set_mode(["joints","target","velocity"].index(args.mode))
    app=App(state,hidden=args.snapshot is not None)
    try:
        if args.snapshot:
            app.snapshot(args.snapshot)
            print(f"Saved {args.snapshot}")
            return
        started=previous=time.perf_counter()
        while not glfw.window_should_close(app.window):
            now=time.perf_counter()
            state.tick(min(now-previous,.04))
            previous=now
            app.render()
            glfw.swap_buffers(app.window)
            glfw.poll_events()
            if args.duration and now-started>=args.duration:
                break
    finally:
        app.close()


if __name__=="__main__":
    main()
