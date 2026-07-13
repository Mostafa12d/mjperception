import mujoco as mj
import mujoco_viewer
import rospy
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge
import tf 

import time
import cv2
import threading
import sys

import numpy as np
import matplotlib.pyplot as plt

from aero_force_v2 import aero, nWagner
from Controller_PID_v1 import *
from utility_functions.rotation_transformations import *
import pandas as pd
import os
import rospkg

# Global control variables
control_commands = {
    'mode': 'pid',  # 'pid' or 'manual'
    'x_setpoint': 0.0,
    'y_setpoint': 0.0,
    'z_setpoint': 0.5,
    'yaw_setpoint': 0.0,
    'motor1': 0.0,
    'motor2': 0.0,
    'motor3': 0.0,
    'motor4': 0.0,
    'motor5': 0.0,
    'motor6': 0.0
}

accel_bias = np.random.normal(0, 0.05, 3)  # Random bias for x, y, z
gyro_bias = np.random.normal(0, 0.002, 3)  # Random bias for roll, pitch, yaw

def terminal_input_thread():
    """Handle terminal input in a separate thread"""
    global control_commands
    
    print("\n=== FLAPPY CONTROL INTERFACE ===")
    print("Commands:")
    print("  mode <pid|manual>     - Switch control mode")
    print("  pid <x> <y> <z> <yaw> - Set PID setpoints (x,y,z in meters, yaw in degrees)")
    print("  motor <1-6> <value>   - Set individual motor (0.0-1.0) in manual mode")
    print("  motors <m1> <m2> <m3> <m4> <m5> <m6> - Set all motors at once")
    print("  status                - Show current status")
    print("  help                  - Show this help")
    print("  quit                  - Exit simulation")
    print("\nExamples:")
    print("  mode pid")
    print("  pid 1.0 0.5 1.0 45")
    print("  mode manual")
    print("  motor 1 0.3")
    print("  motors 0.3 0.3 0.3 0.3 0.1 0.1")
    print("\nEnter commands:")
    
    while True:
        try:
            cmd = input("> ").strip().lower().split()
            if not cmd:
                continue
                
            if cmd[0] == 'quit' or cmd[0] == 'exit':
                print("Exiting simulation...")
                break
                
            elif cmd[0] == 'mode':
                if len(cmd) > 1 and cmd[1] in ['pid', 'manual']:
                    control_commands['mode'] = cmd[1]
                    print(f"Mode set to: {cmd[1]}")
                else:
                    print("Usage: mode <pid|manual>")
                    
            elif cmd[0] == 'pid':
                if len(cmd) == 5:
                    try:
                        control_commands['x_setpoint'] = float(cmd[1])
                        control_commands['y_setpoint'] = float(cmd[2])
                        control_commands['z_setpoint'] = float(cmd[3])
                        control_commands['yaw_setpoint'] = np.deg2rad(float(cmd[4]))
                        print(f"PID setpoints: x={cmd[1]}, y={cmd[2]}, z={cmd[3]}, yaw={cmd[4]}°")
                    except ValueError:
                        print("Invalid numbers. Usage: pid <x> <y> <z> <yaw_deg>")
                else:
                    print("Usage: pid <x> <y> <z> <yaw_deg>")
                    
            elif cmd[0] == 'motor':
                if len(cmd) == 3:
                    try:
                        motor_num = int(cmd[1])
                        motor_val = float(cmd[2])
                        if 1 <= motor_num <= 6 and 0.0 <= motor_val <= 1.0:
                            control_commands[f'motor{motor_num}'] = motor_val
                            print(f"Motor {motor_num} set to: {motor_val}")
                        else:
                            print("Motor number must be 1-6, value must be 0.0-1.0")
                    except ValueError:
                        print("Invalid numbers. Usage: motor <1-6> <0.0-1.0>")
                else:
                    print("Usage: motor <1-6> <0.0-1.0>")
                    
            elif cmd[0] == 'motors':
                if len(cmd) == 7:
                    try:
                        values = [float(x) for x in cmd[1:]]
                        if all(0.0 <= v <= 1.0 for v in values):
                            for i, val in enumerate(values, 1):
                                control_commands[f'motor{i}'] = val
                            print(f"All motors set: {values}")
                        else:
                            print("All values must be between 0.0 and 1.0")
                    except ValueError:
                        print("Invalid numbers. Usage: motors <m1> <m2> <m3> <m4> <m5> <m6>")
                else:
                    print("Usage: motors <m1> <m2> <m3> <m4> <m5> <m6>")
                    
            elif cmd[0] == 'status':
                print(f"\nCurrent Status:")
                print(f"  Mode: {control_commands['mode']}")
                if control_commands['mode'] == 'pid':
                    print(f"  PID Setpoints: x={control_commands['x_setpoint']:.2f}, y={control_commands['y_setpoint']:.2f}, z={control_commands['z_setpoint']:.2f}, yaw={np.rad2deg(control_commands['yaw_setpoint']):.1f}°")
                else:
                    print(f"  Motor Values: {[control_commands[f'motor{i}'] for i in range(1,7)]}")
                    
            elif cmd[0] == 'help':
                print("\nCommands:")
                print("  mode <pid|manual>     - Switch control mode")
                print("  pid <x> <y> <z> <yaw> - Set PID setpoints")
                print("  motor <1-6> <value>   - Set individual motor")
                print("  motors <m1> <m2> <m3> <m4> <m5> <m6> - Set all motors")
                print("  status                - Show current status")
                print("  quit                  - Exit simulation")
                
            else:
                print(f"Unknown command: {cmd[0]}. Type 'help' for commands.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def apply_flight_control(data, controller):
    """Apply control commands from terminal input"""
    global control_commands
    
    if control_commands['mode'] == 'manual':
        # Manual mode - apply motor commands directly
        data.actuator("Motor1").ctrl[0] = control_commands['motor1']
        data.actuator("Motor2").ctrl[0] = control_commands['motor2']
        data.actuator("Motor3").ctrl[0] = control_commands['motor3']
        data.actuator("Motor4").ctrl[0] = control_commands['motor4']
        data.actuator("Motor5").ctrl[0] = control_commands['motor5']
        data.actuator("Motor6").ctrl[0] = control_commands['motor6']
    else:
        # PID mode - update controller setpoints
        controller.x_d = control_commands['x_setpoint']
        controller.y_d = control_commands['y_setpoint']
        controller.z_d = control_commands['z_setpoint']
        controller.yaw_d = control_commands['yaw_setpoint']
        # PID controller handles motor commands automatically

def publish_imu(pub, data):
    imu_msg = Imu()
    # Orientation (quaternion)
    imu_msg.orientation.x = data.sensordata[4]
    imu_msg.orientation.y = data.sensordata[5]
    imu_msg.orientation.z = data.sensordata[6]
    imu_msg.orientation.w = data.sensordata[3]
    # Angular velocity (body frame)
    # Angular velocity with bias (body frame) - sensor data [10:13]
    imu_msg.angular_velocity.x = data.sensordata[10] + gyro_bias[0]
    imu_msg.angular_velocity.y = data.sensordata[11] + gyro_bias[1]
    imu_msg.angular_velocity.z = data.sensordata[12] + gyro_bias[2]
    # Linear acceleration (placeholder)
    imu_msg.linear_acceleration.x = data.sensordata[13] #+ accel_bias[0]
    imu_msg.linear_acceleration.y = data.sensordata[14] #+ accel_bias[1]
    imu_msg.linear_acceleration.z = data.sensordata[15] #+ accel_bias[2]
    pub.publish(imu_msg)

def publish_camera(pub, bridge, renderer, data):
    """Publish camera image using offscreen renderer"""
    renderer.update_scene(data, camera = "onboard_camera")#camera="onboard_camera")
    img = renderer.render()
    ros_img = bridge.cv2_to_imgmsg(img, encoding="rgb8")
    pub.publish(ros_img)

def publish_tf(tf_broadcaster, data):
    position = data.sensordata[0:3]
    orientation = data.sensordata[3:7]  # [w, x, y, z] format
    
    # Convert to ROS format [x, y, z, w]
    qx, qy, qz, qw = orientation[1], orientation[2], orientation[3], orientation[0]
    
    tf_broadcaster.sendTransform(
        (position[0], position[1], position[2]),
        (qx, qy, qz, qw),
        rospy.Time.now(),
        "base_link",
        "odom"
    )

def get_bodyIDs(body_list):
    bodyID_dic = {}
    jntID_dic = {}
    posID_dic = {}
    jvelID_dic = {}
    for bodyName in body_list:
        try:
            mjID = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, bodyName)
            jntID = model.body_jntadr[mjID]
            jvelID = model.body_dofadr[mjID]
            posID = model.jnt_qposadr[jntID]
            bodyID_dic[bodyName] = mjID
            jntID_dic[bodyName] = jntID
            posID_dic[bodyName] = posID
            jvelID_dic[bodyName] = jvelID
        except:
            print(f"Warning: Body '{bodyName}' not found")
    return bodyID_dic, jntID_dic, posID_dic, jvelID_dic

def get_jntIDs(jnt_list):
    jointID_dic = {}
    for jointName in jnt_list:
        try:
            jointID = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jointName)
            jointID_dic[jointName] = jointID
        except:
            print(f"Warning: Joint '{jointName}' not found")
    return jointID_dic

def original_states(model, data):
    xd = np.array([0.0] * 22)
    
    # Handle wing joint data if available
    if "L3" in posID_dic and "L7" in posID_dic:
        xd[0] = data.qpos[posID_dic["L3"]] + np.deg2rad(11.345825599281223)
        xd[1] = data.qpos[posID_dic["L7"]] - np.deg2rad(27.45260202) + xd[0]
        xd[5] = data.qvel[jvelID_dic["L3"]]
        xd[6] = data.qvel[jvelID_dic["L7"]]
    
    xd[2:5] = data.sensordata[0:3]
    xd[7:10] = data.sensordata[7:10]
    xd[10:13] = data.sensordata[10:13]

    if np.linalg.norm(data.sensordata[3:7]) == 0:
        data.sensordata[3:7] = [1,0,0,0]
    R_body = quat2rot(data.sensordata[3:7])
    xd[13:23] = list(np.transpose(R_body).flatten())
    return xd, R_body

""" ------------------------SIMULATION--------------------------------------- """
# Create ROS node and publishers
rospy.init_node('flappy_sim_publisher')
cam_pub = rospy.Publisher('/flappy/camera/image_raw', Image, queue_size=10)
imu_pub = rospy.Publisher('/flappy/imu', Imu, queue_size=10)
tf_broadcaster = tf.TransformBroadcaster()
bridge = CvBridge()

rospack = rospkg.RosPack()
package_path = rospack.get_path('ros_flappy_sim')
csv_path = os.path.join(package_path, 'src', 'JointAngleData.csv')
xml_path = os.path.join(package_path, 'worlds', 'Flappy_v10_weld.xml')

# MuJoCo setup
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# Create offscreen renderer only (no viewer window)
camera_renderer = mj.Renderer(model, height=480, width=640)

# Get body/joint IDs
body_list = ["Guard", "Core", "L3", "L7", "L3R", "L7R"]
joint_list = ['J5', 'J6', 'J5R', 'J6R']
bodyID_dic, jntID_dic, posID_dic, jvelID_dic = get_bodyIDs(body_list)
jID_dic = get_jntIDs(joint_list)

# Wing flapping data
flap_freq = 6
Angle_data = pd.read_csv(csv_path, header=None)
J5_m = Angle_data.loc[:, 0]
J6_m = Angle_data.loc[:, 1]
J5v_m = flap_freq/2 * Angle_data.loc[:, 2]
J6v_m = flap_freq/2 * Angle_data.loc[:, 3]
t_m = np.linspace(0, 1.0/flap_freq, num=len(J5_m))

# Initialize controller
controller = Controller(model, data)
mj.set_mjcb_control(controller.Control)

# Simulation parameters
dt = 1e-3
model.opt.timestep = dt
simend = 60
xa = np.zeros(3 * nWagner)

# Data logging
SimTime = []
ua_ = []
JointAng = [[], []]
JointAng_ref = [[], []]
JointVel = [[], []]
JointVel_ref = [[], []]

# Start terminal input thread
input_thread = threading.Thread(target=terminal_input_thread, daemon=True)
input_thread.start()

n_steps = int(simend / dt)
publish_interval = 10

try:
    t_start = time.time()
    for i in range(n_steps):
        start_time_loop = time.time()

        # Aerodynamics calculation
        if i % 1 == 0:
            xd, R_body = original_states(model, data)
            
            J5v_d = np.interp(data.time, t_m, J5v_m, period=1.0 / flap_freq)
            J6v_d = np.interp(data.time, t_m, J6v_m, period=1.0 / flap_freq)
            xd[5] = J5v_d
            xd[6] = J6v_d

            fa, ua, xd = aero(xd, R_body, xa)

            # Publish ROS topics
            if not rospy.is_shutdown() and (i % publish_interval == 0):
                publish_camera(cam_pub, bridge, camera_renderer, data)
            if not rospy.is_shutdown():
                publish_imu(imu_pub, data)
                publish_tf(tf_broadcaster, data)

            xa = xa + fa * dt

        # Apply aerodynamic forces
        if "L3" in jvelID_dic and "L7" in jvelID_dic:
            data.qfrc_applied[jvelID_dic["L3"]] = ua[0]
            data.qfrc_applied[jvelID_dic["L7"]] = ua[1]
        if "Core" in bodyID_dic:
            data.xfrc_applied[bodyID_dic["Core"]] = [*ua[2:5], *ua[5:8]]

        # Wing flapping
        J5_ = np.interp(data.time, t_m, J5_m, period=1.0/flap_freq)
        J6_ = np.interp(data.time, t_m, J6_m, period=1.0/flap_freq)
        J5_d = J5_ - np.deg2rad(11.345825599281223)
        J6_d = J6_ + np.deg2rad(27.45260202) - J5_
        
        try:
            data.actuator("J5_angle").ctrl[0] = J5_d
            data.actuator("J6_angle").ctrl[0] = J6_d
        except:
            pass

        # Apply flight control
        apply_flight_control(data, controller)

        # Simulation step
        mj.mj_step(model, data)

        # Data logging
        SimTime.append(data.time)
        JointAng_ref[0].append(J5_)
        JointAng_ref[1].append(J6_)
        
        if "L3" in posID_dic and "L7" in posID_dic:
            J5 = data.qpos[posID_dic["L3"]] + np.deg2rad(11.345825599281223)
            J6 = data.qpos[posID_dic["L7"]] - np.deg2rad(27.45260202) + J5
            JointAng[0].append(J5)
            JointAng[1].append(J6)
            JointVel[0].append(data.qvel[jvelID_dic["L3"]])
            JointVel[1].append(data.qvel[jvelID_dic["L7"]])
        else:
            JointAng[0].append(0.0)
            JointAng[1].append(0.0)
            JointVel[0].append(0.0)
            JointVel[1].append(0.0)

        J5v_d = np.interp(data.time, t_m, J5v_m, period=1.0 / flap_freq)
        J6v_d = np.interp(data.time, t_m, J6v_m, period=1.0 / flap_freq)
        JointVel_ref[0].append(J5v_d)
        JointVel_ref[1].append(J6v_d)

        # No rendering - offscreen only for ROS

        # Timing
        elapsed = time.time() - start_time_loop
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        if rospy.is_shutdown():
            break

except KeyboardInterrupt:
    print("\nSimulation interrupted by user.")

finally:
    camera_renderer.close()
    print("Simulation ended and resources released.")

# Results
total_time = time.time() - t_start
print(f"--- Total Time: {total_time:.2f} seconds ---")
print(f"--- Total realtime factor: {simend/total_time:.2f}x ---")

# Plotting
plt.figure()
plt.subplot(2,1,1)
plt.title("Joint Angle (MuJoCo vs. Matlab)")
plt.plot(SimTime, np.rad2deg(JointAng_ref[0]), 'b--', label='J5 ref')
plt.plot(SimTime, np.rad2deg(JointAng[0]), 'b', label='J5 real')
plt.xlabel('Time, t (s)')
plt.ylabel('J5 Angle, θ_5 (deg)')
plt.legend()

plt.subplot(2,1,2)
plt.plot(SimTime, np.rad2deg(JointAng_ref[1]), 'g--', label='J6 ref')
plt.plot(SimTime, np.rad2deg(JointAng[1]), 'g', label='J6 real')
plt.xlabel('Time, t (s)')
plt.ylabel('J6 Angle, θ_6 (deg)')
plt.legend()

plt.show()
