#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.duration import Duration
from rcl_interfaces.msg import SetParametersResult

from std_msgs.msg import Float64, Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3Stamped, TransformStamped
from visualization_msgs.msg import Marker

import tf2_ros

# --- import simulators ---
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from gas_dispersion_simulator import DispersionSimulator
from gas_sensor_simulator import SensorSimulator


def _rot_only_vector_to_target(tf_buffer: tf2_ros.Buffer, target: str, msg: Vector3Stamped):
    """Rotate a stamped vector into target frame (no translation)."""
    try:
        src = msg.header.frame_id or target
        tf: TransformStamped = tf_buffer.lookup_transform(target, src, rclpy.time.Time())
        q = tf.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
        ], dtype=float)
        v = np.array([msg.vector.x, msg.vector.y, msg.vector.z], dtype=float)
        vb = R @ v
        return float(vb[0]), float(vb[1]), float(vb[2])
    except Exception:
        return float(msg.vector.x), float(msg.vector.y), float(msg.vector.z)


class GasSimulatorNode(Node):
    """
    粒子法动态羽流（/gas_plume） + 传感器卷积输出（/gas_sim） + 源点 Marker。
    统一坐标：world。位置优先 TF，否则 /robot_0/odom。
    支持热更新：
      - 源点：source_xyz / source_x,y,z
      - 风向：wind_dir_xyz / wind_dir_x,y,z（自动归一化）
      - 风速：wind_speed
      - 扩散强度：diffusion_speed
    """

    def __init__(self):
        super().__init__('gas_simulator_node')

        # ---------------- Parameters ----------------
        self.declare_parameter('world_frame', 'world')

        # 位置来源（TF优先）
        self.declare_parameter('use_tf', True)
        self.declare_parameter('ee_frame', 'panda_link8')
        self.declare_parameter('odom_topic', '/robot_0/odom')

        # 风向话题（Vector3Stamped）
        self.declare_parameter('wind_topic', '/anemometer_a/wind')

        # 主循环频率
        self.declare_parameter('update_hz', 5.0)

        # 源点 marker
        self.declare_parameter('publish_source_marker', True)
        self.declare_parameter('marker_scale', 0.06)

        # 热更新参数：源点（world）
        self.declare_parameter('source_xyz', [0.0, 0.0, 0.0])
        self.declare_parameter('source_x', 0.0)
        self.declare_parameter('source_y', 0.0)
        self.declare_parameter('source_z', 0.0)

        # 热更新参数：风向（单位向量，会自动归一化）
        self.declare_parameter('wind_dir_xyz', [-1.0, 0.0, 0.0])
        self.declare_parameter('wind_dir_x', -1.0)
        self.declare_parameter('wind_dir_y', 0.0)
        self.declare_parameter('wind_dir_z', 0.0)

        # 热更新参数：风速（标量）
        self.declare_parameter('wind_speed', 0.8)  # m/s

        # 热更新参数：扩散强度（标量）
        self.declare_parameter('diffusion_speed', 0.8)  # m/s

        # 读取参数
        self.world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
        self.use_tf = bool(self.get_parameter('use_tf').value)
        self.ee_frame = self.get_parameter('ee_frame').get_parameter_value().string_value
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.wind_topic = self.get_parameter('wind_topic').get_parameter_value().string_value
        self.update_hz = float(self.get_parameter('update_hz').value)
        self.publish_source_marker = bool(self.get_parameter('publish_source_marker').value)
        self.marker_scale = float(self.get_parameter('marker_scale').value)

        # Fallback 位置（如果 TF 或里程计未到）
        self.robot_x = 5.0
        self.robot_y = 0.0
        self.robot_z = 0.0

        # Simulators
        self.dispersion_simulator = DispersionSimulator()
        self.sensor_simulator = SensorSimulator()
        self.sensor_simulator.set_delta_t(1.0 / max(self.update_hz, 1e-6))

        # ---- 启动时从模拟器读取参数并同步 ----
        sx, sy, sz = self.dispersion_simulator.get_source()
        dir_x, dir_y, dir_z = self.dispersion_simulator.get_wind_dir()
        wind_spd = self.dispersion_simulator.get_wind_speed()
        diff_spd = self.dispersion_simulator.get_diffusion_speed()

        # 同步成参数
        self.set_parameters([
            Parameter('source_x', value=sx),
            Parameter('source_y', value=sy),
            Parameter('source_z', value=sz),
            Parameter('source_xyz', value=[sx, sy, sz]),
            Parameter('wind_dir_x', value=dir_x),
            Parameter('wind_dir_y', value=dir_y),
            Parameter('wind_dir_z', value=dir_z),
            Parameter('wind_dir_xyz', value=[dir_x, dir_y, dir_z]),
            Parameter('wind_speed', value=wind_spd),
            Parameter('diffusion_speed', value=diff_spd),
        ])

        # Pubs
        self.pub_plume = self.create_publisher(PointCloud2, 'gas_plume', 2)
        self.pub_gas = self.create_publisher(Float64, 'gas_sim', 10)
        self.pub_marker = self.create_publisher(Marker, 'gas_source_marker', 1)

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        _listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subs
        self.sub_wind = self.create_subscription(Vector3Stamped, self.wind_topic, self.wind_callback, 10)
        if not self.use_tf:
            self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.position_callback, 10)
        else:
            self.sub_odom = None

        # Timer
        self.timer = self.create_timer(1.0 / max(self.update_hz, 1e-3), self.timer_callback)

        # 热更新回调
        self._in_param_callback = False  # 防止递归
        self.add_on_set_parameters_callback(self._on_param_change)

        self.get_logger().info(
            f'nodegas(particles) ready | world={self.world_frame} use_tf={self.use_tf} '
            f'ee={self.ee_frame} odom={self.odom_topic} wind={self.wind_topic} hz={self.update_hz} '
            f'| src=({sx:.3f},{sy:.3f},{sz:.3f}) '
            f'wind_dir=({dir_x:.3f},{dir_y:.3f},{dir_z:.3f}) wind_speed={wind_spd:.3f}m/s '
            f'diff={diff_spd:.3f}m/s'
        )

    # -------- param callback ----------
    def _on_param_change(self, params):
        """
        支持热更新：
          - source_xyz / source_x,y,z  → set_source()
          - wind_dir_xyz / wind_dir_x,y,z → set_wind_dir()
          - wind_speed → set_wind_speed()
          - diffusion_speed → set_diffusion_speed()
          - update_hz → 改主循环 & 同步传感器步长
        """
        # 防止递归
        if self._in_param_callback:
            return SetParametersResult(successful=True)
        self._in_param_callback = True
        
        try:
            sx = sy = sz = None
            wind_dir_x = wind_dir_y = wind_dir_z = None
            wind_spd = None
            diff_spd = None

            for p in params:
                # 基础参数
                if p.name == 'world_frame' and p.type_ == Parameter.Type.STRING:
                    self.world_frame = p.value
                elif p.name == 'use_tf' and p.type_ == Parameter.Type.BOOL:
                    self.use_tf = bool(p.value)
                elif p.name == 'ee_frame' and p.type_ == Parameter.Type.STRING:
                    self.ee_frame = p.value
                elif p.name == 'odom_topic' and p.type_ == Parameter.Type.STRING:
                    self.odom_topic = p.value
                elif p.name == 'wind_topic' and p.type_ == Parameter.Type.STRING:
                    self.wind_topic = p.value
                elif p.name == 'update_hz' and p.type_ in (Parameter.Type.INTEGER, Parameter.Type.DOUBLE):
                    hz = float(p.value)
                    if hz > 0.0:
                        self.timer.cancel()
                        self.timer = self.create_timer(1.0/hz, self.timer_callback)
                        self.update_hz = hz
                        self.sensor_simulator.set_delta_t(1.0 / hz)
                elif p.name == 'publish_source_marker' and p.type_ == Parameter.Type.BOOL:
                    self.publish_source_marker = bool(p.value)
                elif p.name == 'marker_scale' and p.type_ in (Parameter.Type.INTEGER, Parameter.Type.DOUBLE):
                    self.marker_scale = float(p.value)

                # 源点：数组形式
                elif p.name == 'source_xyz' and p.type_ == Parameter.Type.DOUBLE_ARRAY:
                    arr = list(p.value)
                    if len(arr) >= 3:
                        sx, sy, sz = float(arr[0]), float(arr[1]), float(arr[2])
                        self.set_parameters([
                            Parameter('source_x', value=sx),
                            Parameter('source_y', value=sy),
                            Parameter('source_z', value=sz),
                        ])
                # 源点：标量形式
                elif p.name == 'source_x' and p.type_ in (Parameter.Type.INTEGER, Parameter.Type.DOUBLE):
                    sx = float(p.value)
                elif p.name == 'source_y' and p.type_ in (Parameter.Type.INTEGER, Parameter.Type.DOUBLE):
                    sy = float(p.value)
                elif p.name == 'source_z' and p.type_ in (Parameter.Type.INTEGER, Parameter.Type.DOUBLE):
                    sz = float(p.value)

                # 风向：数组形式
                elif p.name == 'wind_dir_xyz' and p.type_ == Parameter.Type.DOUBLE_ARRAY:
                    arr = list(p.value)
                    if len(arr) >= 3:
                        wind_dir_x, wind_dir_y, wind_dir_z = float(arr[0]), float(arr[1]), float(arr[2])
                        self.set_parameters([
                            Parameter('wind_dir_x', value=wind_dir_x),
                            Parameter('wind_dir_y', value=wind_dir_y),
                            Parameter('wind_dir_z', value=wind_dir_z),
                        ])
                # 风向：标量形式
                elif p.name == 'wind_dir_x' and p.type_ in (Parameter.Type.INTEGER, Parameter.Type.DOUBLE):
                    wind_dir_x = float(p.value)
                elif p.name == 'wind_dir_y' and p.type_ in (Parameter.Type.INTEGER, Parameter.Type.DOUBLE):
                    wind_dir_y = float(p.value)
                elif p.name == 'wind_dir_z' and p.type_ in (Parameter.Type.INTEGER, Parameter.Type.DOUBLE):
                    wind_dir_z = float(p.value)

                # 风速（标量）
                elif p.name == 'wind_speed' and p.type_ in (Parameter.Type.INTEGER, Parameter.Type.DOUBLE):
                    wind_spd = float(p.value)

                # 扩散强度（标量）
                elif p.name == 'diffusion_speed' and p.type_ in (Parameter.Type.INTEGER, Parameter.Type.DOUBLE):
                    diff_spd = float(p.value)

            # ---------- 应用源点 ----------
            cur_sx = float(self.get_parameter('source_x').value)
            cur_sy = float(self.get_parameter('source_y').value)
            cur_sz = float(self.get_parameter('source_z').value)
            if sx is None: sx = cur_sx
            if sy is None: sy = cur_sy
            if sz is None: sz = cur_sz
            self.dispersion_simulator.set_source(sx, sy, sz)
            self.set_parameters([
                Parameter('source_xyz', value=[sx, sy, sz]),
            ])

            # ---------- 应用风向 ----------
            cur_dir_x = float(self.get_parameter('wind_dir_x').value)
            cur_dir_y = float(self.get_parameter('wind_dir_y').value)
            cur_dir_z = float(self.get_parameter('wind_dir_z').value)
            if wind_dir_x is None: wind_dir_x = cur_dir_x
            if wind_dir_y is None: wind_dir_y = cur_dir_y
            if wind_dir_z is None: wind_dir_z = cur_dir_z
            self.dispersion_simulator.set_wind_dir(wind_dir_x, wind_dir_y, wind_dir_z)
            # 同步归一化后的值
            norm_x, norm_y, norm_z = self.dispersion_simulator.get_wind_dir()
            self.set_parameters([
                Parameter('wind_dir_x', value=norm_x),
                Parameter('wind_dir_y', value=norm_y),
                Parameter('wind_dir_z', value=norm_z),
                Parameter('wind_dir_xyz', value=[norm_x, norm_y, norm_z]),
            ])

            # ---------- 应用风速 ----------
            if wind_spd is not None:
                self.dispersion_simulator.set_wind_speed(wind_spd)

            # ---------- 应用扩散强度 ----------
            if diff_spd is not None:
                self.dispersion_simulator.set_diffusion_speed(diff_spd)

            self.get_logger().info(
                f'[hot-update] src=({sx:.3f},{sy:.3f},{sz:.3f}) '
                f'wind_dir=({norm_x:.3f},{norm_y:.3f},{norm_z:.3f}) '
                f'wind_speed={self.dispersion_simulator.get_wind_speed():.3f}m/s '
                f'diff={self.dispersion_simulator.get_diffusion_speed():.3f}m/s'
            )

            self._in_param_callback = False
            return SetParametersResult(successful=True)
        except Exception as e:
            self._in_param_callback = False
            return SetParametersResult(successful=False, reason=str(e))

    # -------- loop ----------
    def timer_callback(self):
        # 动态羽流
        self.dispersion_simulator.update()

        # plume & marker
        self.publish_pointcloud()
        if self.publish_source_marker:
            self.publish_source_marker_fn()

        # 传感器输出
        self.publish_gas_sensor()

    # -------- subs ----------
    def position_callback(self, msg: Odometry):
        self.robot_x = float(msg.pose.pose.position.x)
        self.robot_y = float(msg.pose.pose.position.y)
        self.robot_z = float(msg.pose.pose.position.z)

    def wind_callback(self, msg: Vector3Stamped):
        """从话题接收风速向量，分解为风向和风速。"""
        vx, vy, vz = msg.vector.x, msg.vector.y, msg.vector.z
        if (msg.header.frame_id or '') and (msg.header.frame_id != self.world_frame):
            vx, vy, vz = _rot_only_vector_to_target(self.tf_buffer, self.world_frame, msg)
        
        # 分解为风向和风速
        speed = math.sqrt(vx**2 + vy**2 + vz**2)
        if speed > 1e-9:
            self.dispersion_simulator.set_wind_dir(vx, vy, vz)
        self.dispersion_simulator.set_wind_speed(speed)

    # -------- pubs ----------
    def publish_gas_sensor(self):
        if self.use_tf:
            try:
                tf: TransformStamped = self.tf_buffer.lookup_transform(
                    self.world_frame, self.ee_frame, rclpy.time.Time())
                p = tf.transform.translation
                px, py, pz = float(p.x), float(p.y), float(p.z)
            except Exception:
                px, py, pz = self.robot_x, self.robot_y, self.robot_z
        else:
            px, py, pz = self.robot_x, self.robot_y, self.robot_z

        c_real = self.dispersion_simulator.evaluate_concentration(px, py, pz)
        y = self.sensor_simulator.update(c_real)

        msg = Float64(); msg.data = float(y)
        self.pub_gas.publish(msg)

    def publish_pointcloud(self):
        nodes = np.vstack((
            self.dispersion_simulator.particle_x,
            self.dispersion_simulator.particle_y,
            self.dispersion_simulator.particle_z
        ))
        color = self.dispersion_simulator.particle_age  # age as color

        fields = [PointField(name=c, offset=4*i, datatype=PointField.FLOAT32, count=1)
                  for i, c in enumerate('xyzc')]
        points = np.vstack((nodes, color.reshape(1, -1))).astype(float).T

        header = Header(); header.frame_id = self.world_frame
        pc2 = point_cloud2.create_cloud(header, fields, points)
        pc2.header.stamp = self.get_clock().now().to_msg()
        self.pub_plume.publish(pc2)

    def publish_source_marker_fn(self):
        sx, sy, sz = self.dispersion_simulator.get_source()

        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'nodegas'; marker.id = 1
        marker.type = Marker.SPHERE; marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = float(sx)
        marker.pose.position.y = float(sy)
        marker.pose.position.z = float(sz)
        marker.scale.x = marker.scale.y = marker.scale.z = float(self.marker_scale)
        marker.color.r = 1.0; marker.color.g = 0.2; marker.color.b = 0.2; marker.color.a = 0.95
        self.pub_marker.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = GasSimulatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()



