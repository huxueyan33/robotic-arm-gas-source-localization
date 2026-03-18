#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gas Seek Bio v9 - 带推进的精细定位SPIRAL

核心改进：
- SPIRAL = DRIFT（小步推进+螺旋摆动）+ PROBE（局部微扰+梯度估计）循环
- 振幅/步长随时间衰减，自然收敛
- 三类退出逻辑：成功收敛 / 信号丢失 / 工程超时
"""

import math
import os
import csv
import time
import datetime
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException

from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from rcl_interfaces.srv import GetParameters
import tf2_ros
from geometry_msgs.msg import TransformStamped


# ------------------ utils ------------------ #
def db10(x: float) -> float:
    x = max(float(x), 1e-12)
    return 10.0 * math.log10(x)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return (v / n) if n > 1e-9 else np.zeros_like(v)


class BaselineEstimator:
    """
    PID 传感器 Baseline 估计：
    - 长/短窗口分位数
    - 候选取较小值（保守）
    - baseline 下降可以快，上升很慢，避免把 plume 当背景
    """

    def __init__(self, baseline_init: float = 3.0,
                 baseline_min: float = 1.0,
                 percentile: float = 0.15):
        self.baseline = baseline_init
        self.baseline_min = baseline_min
        self.percentile = percentile

    def _percentile(self, data, p: float) -> float:
        if not data:
            return self.baseline
        n = len(data)
        k = max(1, int(n * p))
        return sorted(data)[k - 1]

    def update(self, long_hist, short_hist):
        """根据长/短窗口更新 baseline"""
        long_list = list(long_hist)
        short_list = list(short_hist)

        if long_list:
            baseline_slow = self._percentile(long_list, self.percentile)
        else:
            baseline_slow = self.baseline

        if short_list:
            baseline_fast = self._percentile(short_list, self.percentile)
        else:
            baseline_fast = baseline_slow

        baseline_candidate = min(baseline_slow, baseline_fast)

        if baseline_candidate < self.baseline:
            self.baseline = 0.9 * self.baseline + 0.1 * baseline_candidate
        else:
            self.baseline = 0.99 * self.baseline + 0.01 * baseline_candidate

        self.baseline = max(self.baseline, self.baseline_min)
        return self.baseline


class DirectionEstimate:
    """
    CAST 输出的方向估计（侦察兵模式）
    """
    def __init__(self):
        self.reach_dir = 0.0
        self.yaw_dir = 0.0
        self.pitch_dir = 0.0
        
        self.reach_conf = 0.0
        self.yaw_conf = 0.0
        self.pitch_conf = 0.0
        self.confidence = 0.0
        
        self.timestamp = 0.0
        self.valid = False
        
        self.n_left = 0
        self.n_right = 0
        self.n_forward = 0
        self.n_backward = 0
        self.n_up = 0
        self.n_down = 0
        
        self.gradient_lr = 0.0
        self.gradient_fb = 0.0
        self.gradient_ud = 0.0
    
    def age(self, current_time: float) -> float:
        return current_time - self.timestamp
    
    def get_weight(self, current_time: float, decay_time: float = 8.0) -> float:
        if not self.valid:
            return 0.0
        age = self.age(current_time)
        if age > decay_time:
            return 0.0
        return self.confidence * (1.0 - age / decay_time)
    
    def reset(self):
        self.reach_dir = 0.0
        self.yaw_dir = 0.0
        self.pitch_dir = 0.0
        self.reach_conf = 0.0
        self.yaw_conf = 0.0
        self.pitch_conf = 0.0
        self.confidence = 0.0
        self.timestamp = 0.0
        self.valid = False
        self.n_left = self.n_right = 0
        self.n_forward = self.n_backward = 0
        self.n_up = self.n_down = 0
        self.gradient_lr = self.gradient_fb = self.gradient_ud = 0.0
    
    def __repr__(self):
        return (f"DirectionEstimate(reach={self.reach_dir:+.2f}, yaw={self.yaw_dir:+.2f}, "
                f"pitch={self.pitch_dir:+.2f}, conf={self.confidence:.2f}, valid={self.valid}, "
                f"samples=[L{self.n_left}/R{self.n_right}/F{self.n_forward}/B{self.n_backward}])")


class GasSeekBio(Node):

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #
    def __init__(self):
        super().__init__('gas_seek_bio')

        # ---------- 信号 / 阈值 ----------
        self.declare_parameter('rel_db_high', 0.20)
        self.declare_parameter('rel_db_low', 0.10)
        self.declare_parameter('rel_db_mid', 0.15)

        self.declare_parameter('signal_mode', 'delta')
        self.declare_parameter('background_ppb', 50.0)
        self.declare_parameter('sensor_noise', 3.0)
        self.declare_parameter('baseline_min', 1.0)

        self.declare_parameter('delta_low', 50.0)
        self.declare_parameter('delta_mid', 120.0)
        self.declare_parameter('delta_high', 200.0)

        self.declare_parameter('margin_low', 0.15)
        self.declare_parameter('margin_mid', 0.30)
        self.declare_parameter('margin_high', 0.50)

        self.declare_parameter('hit_short_window_s', 1.5)
        self.declare_parameter('hit_long_window_s', 4.0)
        self.declare_parameter('hits_to_enter', 2)
        self.declare_parameter('hits_to_exit', 5)
        self.declare_parameter('surge_min_duration_s', 3.0)

        # ---------- baseline ----------
        self.declare_parameter('baseline_db_init', 50.0)
        self.declare_parameter('baseline_window', 100)
        self.declare_parameter('baseline_percentile', 0.15)
        self.declare_parameter('post_escape_short_window', 80)
        self.declare_parameter('post_escape_short_min_samples', 20)

        # ---------- stop conditions ----------
        self.declare_parameter('stop_mode', 'distance')
        self.declare_parameter('stop_dist_m', 0.17)
        self.declare_parameter('plateau_db', 2.0)
        self.declare_parameter('plateau_hits', 6)
        self.declare_parameter('disable_platform_hold', True)

        # ---------- joints / limits ----------
        self.declare_parameter('joint_names', [
            'panda_joint1', 'panda_joint2', 'panda_joint3',
            'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
        ])
        self.declare_parameter('joint_limits_lower',
                               [-2.9, -1.76, -2.9, -3.07, -2.9, -3.75, -2.9])
        self.declare_parameter('joint_limits_upper',
                               [2.9, 1.76, 2.9, -0.07, 2.9, 3.75, 2.9])

        # ---------- frames ----------
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('ee_frame', 'panda_link8')
        self.declare_parameter('sensor_frame', 'panda_link7')

        # ---------- control ----------
        self.declare_parameter('loop_hz', 5.0)
        self.declare_parameter('traj_duration_sec', 0.6)
        self.declare_parameter('max_step_deg', 5.0)
        self.declare_parameter('cmd_lowpass_alpha', 0.6)
        self.declare_parameter('yaw_step_deg', 6.0)
        self.declare_parameter('pitch_step_deg', 3.0)
        self.declare_parameter('reach_step_deg', 3.0)

        # ---------- SURGE ----------
        self.declare_parameter('probe_yaw_step_deg', 3.0)
        self.declare_parameter('signal_side_window', 6)
        self.declare_parameter('signal_side_eps_db', 5.0)
        self.declare_parameter('progress_eps_db', 8.0)
        self.declare_parameter('progress_no_hits', 6)
        self.declare_parameter('forward_scale', 1.2)
        self.declare_parameter('allow_vertical_probe', True)
        self.declare_parameter('surge_check_interval', 6)
        self.declare_parameter('surge_drop_db', 0.5)

        # ---------- CAST ----------
        self.declare_parameter('cast_probe_yaw_step_deg', 16.0)
        self.declare_parameter('cast_forward_scale', 0.45)
        self.declare_parameter('cast_side_window', 8)
        self.declare_parameter('cast_side_eps_db', 5.0)
        self.declare_parameter('cast_progress_eps_db', 8.0)

        # ---------- CAST 侦察兵参数 ----------
        self.declare_parameter('cast_min_samples_per_direction', 8)
        self.declare_parameter('cast_min_duration_s', 3.0)
        self.declare_parameter('cast_cv_max', 0.8)
        self.declare_parameter('cast_dir_confidence_threshold', 0.25)
        self.declare_parameter('cast_dir_decay_time_s', 40.0)

        # ---------- SPIRAL v9 参数 ----------
        self.declare_parameter('spiral_drift_duration_s', 1.5)
        self.declare_parameter('spiral_probe_duration_s', 2.0)
        self.declare_parameter('spiral_decay_rate', 0.85)
        self.declare_parameter('spiral_min_scale', 0.15)
        self.declare_parameter('spiral_max_time_s', 45.0)
        self.declare_parameter('spiral_converge_grad_thr', 1.5)
        self.declare_parameter('spiral_converge_margin_thr', 5.0)

        # ---------- EXPLORE / stuck ----------
        self.declare_parameter('explore_enabled', True)
        self.declare_parameter('explore_timeout_s', 3.5)
        self.declare_parameter('explore_yaw_step_deg', 10.0)
        self.declare_parameter('explore_pitch_step_deg', 6.0)
        self.declare_parameter('stuck_window_s', 6.0)
        self.declare_parameter('stuck_min_improve_db', 0.14)
        self.declare_parameter('stuck_max_switches', 4)
        self.declare_parameter('stuck_cooldown_s', 8.0)

        # ---------- timeout ----------
        self.declare_parameter('timeout_s', 120.0)

        # ---------- logging ----------
        self.declare_parameter('log_dir', os.path.expanduser('~/gas_runs'))
        self.declare_parameter('start_zone', 'unknown')
        self.declare_parameter('run_id', 0)
        self.declare_parameter('point_id', 'P0')
        self.declare_parameter('trial_num', 0)
        self.declare_parameter('gas_state', 'balanced')
        self.declare_parameter('intermittency_index', 1.0)

        # ---------- gas topic ----------
        self.declare_parameter('gas_topic', '/gas_sim')

        # ---------- 读取参数 ----------
        p = lambda n: self.get_parameter(n).value

        self.rel_db_high = float(p('rel_db_high'))
        self.rel_db_low = float(p('rel_db_low'))
        self.rel_db_mid = float(p('rel_db_mid'))

        self.signal_mode = str(p('signal_mode'))
        self.background_ppb = float(p('background_ppb'))
        self.sensor_noise = float(p('sensor_noise'))
        self.baseline_min = float(p('baseline_min'))

        self.delta_low = float(p('delta_low'))
        self.delta_mid = float(p('delta_mid'))
        self.delta_high = float(p('delta_high'))

        self.margin_low = float(p('margin_low'))
        self.margin_mid = float(p('margin_mid'))
        self.margin_high = float(p('margin_high'))

        self.baseline_init_const = float(p('baseline_db_init'))
        self.baseline = self.baseline_init_const
        self.baseline_window = int(p('baseline_window'))
        self.baseline_percentile = float(p('baseline_percentile'))
        self.post_escape_short_window = int(p('post_escape_short_window'))
        self.post_escape_short_min_samples = int(p('post_escape_short_min_samples'))

        self.stop_mode = str(p('stop_mode'))
        self.stop_dist_m = float(p('stop_dist_m'))
        self.plateau_db = float(p('plateau_db'))
        self.plateau_hits = int(p('plateau_hits'))
        self.disable_platform_hold = bool(p('disable_platform_hold'))

        self.joint_names = list(p('joint_names'))
        self.lower = np.array(p('joint_limits_lower'), dtype=float)
        self.upper = np.array(p('joint_limits_upper'), dtype=float)

        self.world = str(p('world_frame'))
        self.ee = str(p('ee_frame'))
        self.sensor_frame = str(p('sensor_frame'))

        self.loop_hz = float(p('loop_hz'))
        self.dt = 1.0 / self.loop_hz
        self.traj_duration = float(p('traj_duration_sec'))
        self.max_step = math.radians(float(p('max_step_deg')))
        self.alpha = float(p('cmd_lowpass_alpha'))
        self.yaw_step = math.radians(float(p('yaw_step_deg')))
        self.pitch_step = math.radians(float(p('pitch_step_deg')))
        self.reach_step = math.radians(float(p('reach_step_deg')))

        self.hit_short_window_s = float(p('hit_short_window_s'))
        self.hit_long_window_s = float(p('hit_long_window_s'))
        self.hit_short_window = max(1, int(self.hit_short_window_s * self.loop_hz))
        self.hit_long_window = max(1, int(self.hit_long_window_s * self.loop_hz))
        self.hits_to_enter = int(p('hits_to_enter'))
        self.hits_to_exit = int(p('hits_to_exit'))
        self.surge_min_duration_s = float(p('surge_min_duration_s'))

        self.probe_yaw_step = math.radians(float(p('probe_yaw_step_deg')))
        self.signal_side_window = int(p('signal_side_window'))
        self.signal_side_eps = float(p('signal_side_eps_db'))
        self.progress_eps = float(p('progress_eps_db'))
        self.progress_no_hits = int(p('progress_no_hits'))
        self.forward_scale = float(p('forward_scale'))
        self.allow_vertical_probe = bool(p('allow_vertical_probe'))
        self.surge_check_interval = int(p('surge_check_interval'))
        self.surge_drop_db = float(p('surge_drop_db'))

        if self.signal_mode == 'ratio':
            if self.signal_side_eps > 1.0:
                self.signal_side_eps *= 0.01
            if self.progress_eps > 1.0:
                self.progress_eps *= 0.01
        else:
            if self.signal_side_eps < 1.0:
                self.signal_side_eps = 5.0
            if self.progress_eps < 1.0:
                self.progress_eps = 8.0

        self.cast_probe_yaw_step = math.radians(float(p('cast_probe_yaw_step_deg')))
        self.cast_forward_scale = float(p('cast_forward_scale'))
        self.cast_side_window = int(p('cast_side_window'))
        self.cast_side_eps = float(p('cast_side_eps_db'))
        self.cast_progress_eps = float(p('cast_progress_eps_db'))

        if self.signal_mode == 'ratio':
            if self.cast_side_eps > 1.0:
                self.cast_side_eps *= 0.01
            if self.cast_progress_eps > 1.0:
                self.cast_progress_eps *= 0.01
        else:
            if self.cast_side_eps < 1.0:
                self.cast_side_eps = 5.0
            if self.cast_progress_eps < 1.0:
                self.cast_progress_eps = 8.0

        self.cast_min_samples = int(p('cast_min_samples_per_direction'))
        self.cast_min_duration_s = float(p('cast_min_duration_s'))
        self.cast_cv_max = float(p('cast_cv_max'))
        self.cast_dir_confidence_threshold = float(p('cast_dir_confidence_threshold'))
        self.cast_dir_decay_time_s = float(p('cast_dir_decay_time_s'))

        # SPIRAL v9 参数
        self.spiral_drift_duration = float(p('spiral_drift_duration_s'))
        self.spiral_probe_duration = float(p('spiral_probe_duration_s'))
        self.spiral_decay_rate = float(p('spiral_decay_rate'))
        self.spiral_min_scale = float(p('spiral_min_scale'))
        self.spiral_max_time = float(p('spiral_max_time_s'))
        self.spiral_converge_grad_thr = float(p('spiral_converge_grad_thr'))
        self.spiral_converge_margin_thr = float(p('spiral_converge_margin_thr'))

        self.explore_enabled = bool(p('explore_enabled'))
        self.explore_timeout = float(p('explore_timeout_s'))
        self.explore_yaw = math.radians(float(p('explore_yaw_step_deg')))
        self.explore_pitch = math.radians(float(p('explore_pitch_step_deg')))
        self.stuck_window_s = float(p('stuck_window_s'))
        self.stuck_min_improve_db = float(p('stuck_min_improve_db'))
        self.stuck_max_switches = int(p('stuck_max_switches'))
        self.stuck_cooldown_s = float(p('stuck_cooldown_s'))

        self.gas_topic = str(p('gas_topic'))

        # ---------- TF ----------
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---------- subs / pubs ----------
        self.db_hist = deque(maxlen=self.baseline_window)
        self.db_hist_new = deque(maxlen=self.post_escape_short_window)
        self.baseline_slow = self.baseline
        self.baseline_fast = self.baseline
        self.baseline_epoch = 0
        self.baseline_warmup_steps = 0

        self.baseline_est = BaselineEstimator(
            baseline_init=self.baseline_init_const,
            baseline_min=self.baseline_min,
            percentile=self.baseline_percentile
        )

        self.sub_gas = self.create_subscription(
            Float64, self.gas_topic, self.on_gas, 10
        )
        self.pub_traj = self.create_publisher(
            JointTrajectory, '/panda_arm_controller/joint_trajectory', 10
        )

        self._have_js = False
        self._q_cur = np.zeros(len(self.joint_names))
        self._q_cmd = None
        self.create_subscription(JointState, '/joint_states', self.on_joint_states, 10)

        # ---------- source (仿真用) ----------
        self.src = np.array([0.8, 0.0, 0.8], dtype=float)
        self.param_client = self.create_client(GetParameters, '/gas_simulator_node/get_parameters')
        self.create_timer(3.0, self.update_source)

        # ---------- 状态机 ----------
        self.state = 'CAST'
        self.cast_hits = 0
        self.surge_hits = 0

        # ---------- SURGE 状态 ----------
        self._surge_start_time = None
        self._last_dq = None
        self._surge_neg_cnt = 0
        self._surge_entry_baseline = None  # 进入SURGE时的baseline
        self._surge_margin_hist = deque(maxlen=20)  # margin趋势检测
        self._surge_max_margin = -1e9  # 峰值margin检测

        # ---------- SPIRAL 状态 ----------
        self._spiral_entry_baseline = None  # 进入SPIRAL时的baseline

        # ---------- CAST 侦察兵状态 ----------
        self._direction_estimate = DirectionEstimate()
        self._cast_start_time = None
        self._cast_ready_for_surge = False
        self._cast_samples_left = []
        self._cast_samples_right = []
        self._cast_samples_forward = []
        self._cast_samples_backward = []
        self._cast_samples_up = []
        self._cast_samples_down = []

        # ---------- SPIRAL v9 状态 ----------
        self._spiral_q0 = None
        self._spiral_t0 = None
        self._spiral_substate = 'DRIFT'  # 'DRIFT' | 'PROBE' | 'HOLD_FINAL'
        self._spiral_substate_t0 = None
        
        # 推进方向
        self._spiral_drift_dir = np.array([0.0, 0.8, 0.0])  # [yaw, reach, pitch]
        
        # 衰减控制
        self._spiral_drift_scale = 1.0
        self._spiral_probe_scale = 1.0
        
        # 采样记录
        self._spiral_probe_samples = []
        self._spiral_probe_best_offset = np.zeros(2)  # [j5_offset, j6_offset]
        
        # 收敛检测
        self._spiral_best_margin_hist = deque(maxlen=5)
        self._spiral_gradient_hist = deque(maxlen=5)
        
        # 退出计数
        self._spiral_fail_count = 0
        self._spiral_no_improve_count = 0
        self._spiral_enter_count = 0
        self._spiral_exit_count = 0
        
        # 最佳记录
        self._spiral_best_margin = 0.0
        self._spiral_best_q = None

        # ---------- internal caches ----------
        self._side_left = deque(maxlen=self.signal_side_window)
        self._side_right = deque(maxlen=self.signal_side_window)
        self._side_forward = deque(maxlen=self.signal_side_window)
        self._side_backward = deque(maxlen=self.signal_side_window)
        self._side_up = deque(maxlen=self.signal_side_window)
        self._side_down = deque(maxlen=self.signal_side_window)
        self._cast_left = deque(maxlen=self.cast_side_window)
        self._cast_right = deque(maxlen=self.cast_side_window)
        self._cast_probe_sign = 0
        self._cast_pref_sign = 0
        self._cast_steps_since_core = 0

        self._switch_count = 0
        self._last_gas_raw = 0.0
        self._step_idx = 0

        self._hit_short = deque(maxlen=self.hit_short_window)
        self._hit_long = deque(maxlen=self.hit_long_window)
        self._margin_short = deque(maxlen=self.hit_short_window)
        self._margin_long = deque(maxlen=self.hit_long_window)

        self.plume_phase = 'NO_CONTACT'
        self._prev_plume_phase = 'NO_CONTACT'
        self._hit_state = 0

        self._plume_window_s = 2.0
        self._plume_window_steps = max(1, int(round(self._plume_window_s * self.loop_hz)))
        self._hit_hist_phase = deque(maxlen=self._plume_window_steps)
        self._margin_hist_phase = deque(maxlen=self._plume_window_steps)

        if self.signal_mode == 'ratio':
            self._plume_on_margin = max(self.margin_mid, self.margin_high * 1.1)
            self._plume_off_margin = self.margin_low
        else:
            self._plume_on_margin = max(self.delta_mid * 0.8, self.delta_high * 0.6)
            self._plume_off_margin = self.delta_low * 0.8

        self._plume_core_ratio = 0.55
        self._plume_bg_ratio = 0.05
        self._plume_bg_gap_s = 2.5
        self._plume_candidate_time_s = 1.2
        self._plume_core_time = 0.0

        self._gas_hist_full = []
        self._base_hist_full = []

        self._prev_ee = None
        self._prev_margin = None

        self._ee_hist = deque(maxlen=60)
        self._margin_hist = deque(maxlen=60)

        self._global_core_peak_margin = -1e9
        self._global_core_peak_pos = None

        self._core_center_pos = None
        self._core_center_margin = -1e9
        self._core_radius_min = 0.15
        self._core_radius_max = 0.40

        # ---------- 启动阶段检测 ----------
        self._startup_done = False
        self._startup_phase = 'STATIC'
        self._startup_t0 = None
        self._startup_samples = []
        self._startup_last_db = None
        self._startup_q0 = None
        self._startup_class = None
        self._startup_metrics = {}
        self._startup_static_s = 2.0
        self._startup_micro_s = 2.0
        self._startup_total_s = 4.0

        self.plume_zone = 'near'

        # ---------- SURGE 历史方向 ----------
        self._last_good_yaw_dir = 0.0
        self._last_good_reach_dir = 1.0
        self._last_good_pitch_dir = 0.0
        self._last_margin_surge = None

        # ---------- 源点判定 ----------
        self._src_window_s = 3.0
        self._src_window_steps = max(1, int(round(self._src_window_s * self.loop_hz)))
        self._src_margin_hist = deque(maxlen=self._src_window_steps)
        self._src_hit_hist = deque(maxlen=self._src_window_steps)
        self.source_state = 'NONE'
        self.source_conf = 0.0
        self._src_candidate_t0 = None
        self._src_confirm_t0 = None
        self._source_stop_done = False
        self._src_hold_time = 3.0

        self._src_min_run_time_s = 12.0
        self._src_min_move_dist = 0.05
        self._src_initial_ee = None
        self._src_peak_margin = -1e9
        self._src_peak_margin_pos = None

        # ---------- timeout ----------
        self.timeout_s = float(p('timeout_s'))
        self._summary_written = False
        self._start_ee_pos = None
        self._total_travel_dist = 0.0
        self._prev_ee_for_travel = None

        # ---------- 实验追踪变量（用于鲁棒性测试分析）----------
        # 时间节点
        self._time_first_surge = None
        self._time_first_spiral = None
        self._time_first_confirmed = None
        self._time_first_core = None
        
        # 各状态停留时间
        self._time_in_cast = 0.0
        self._time_in_surge = 0.0
        self._time_in_spiral = 0.0
        self._prev_state_for_time = 'CAST'
        self._prev_state_time = None
        
        # 状态切换计数
        self._num_cast_to_surge = 0
        self._num_surge_to_cast = 0
        self._num_surge_to_spiral = 0
        self._num_spiral_to_surge = 0
        self._num_spiral_to_cast = 0
        
        # 峰值记录
        self._peak_source_conf = 0.0
        self._peak_source_conf_time = None
        
        # 初始距离
        self._init_distance = None

        # ---------- logging ----------
        self.log_dir_base = str(p('log_dir'))
        self.start_zone = str(p('start_zone'))
        self.run_id = int(p('run_id'))
        self.point_id = str(p('point_id'))
        self.trial_num = int(p('trial_num'))
        self.gas_state = str(p('gas_state'))
        self.intermittency_index = float(p('intermittency_index'))
        
        # 直接使用传入的log_dir（runcase已设置好子目录）
        self.log_dir = self.log_dir_base
        os.makedirs(self.log_dir, exist_ok=True)
        
        tstamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.t0 = time.time()
        
        run_id_str = f"{self.run_id:03d}"
        self.log_path = os.path.join(self.log_dir, f"run_{run_id_str}.csv")
        self.log_file = open(self.log_path, "w", newline="")
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow([
            "time_s", "step_idx",
            "gas_raw", "gas_conc", "margin",
            "dist_m",
            "sensor_x", "sensor_y", "sensor_z",
            "src_x", "src_y", "src_z",
            "state", "surge_event", "state_change", "switches",
            "dir_est_reach", "dir_est_yaw", "dir_est_conf",
            "spiral_substate", "spiral_drift_scale", "spiral_probe_scale"
        ])
        self.summary_path = os.path.join(self.log_dir, f"run_{run_id_str}_summary.csv")
        self._last_print_sec = -1

        self.create_timer(self.dt, self.loop)
        self.get_logger().info(
            "gas_seek_bio v9 ready - SPIRAL with DRIFT+PROBE cycle, "
            "adaptive decay, triple exit logic."
        )

    # ------------------------------------------------------------------ #
    # callbacks
    # ------------------------------------------------------------------ #
    def on_joint_states(self, msg: JointState):
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        for n, pos in zip(msg.name, msg.position):
            if n in name_to_idx and n in self.joint_names:
                self._q_cur[self.joint_names.index(n)] = float(pos)
        if not self._have_js:
            self._have_js = True
            self._q_cmd = self._q_cur.copy()
            self.get_logger().info("joint_states received - initialized command pose.")

    def update_source(self):
        if not self.param_client.service_is_ready():
            return
        req = GetParameters.Request()
        req.names = ['source_x', 'source_y', 'source_z']
        fut = self.param_client.call_async(req)
        fut.add_done_callback(self._on_src)

    def _on_src(self, fut):
        try:
            res = fut.result()
            vals = [v.double_value for v in res.values]
        except Exception:
            return
        if len(vals) == 3:
            self.src = np.array(vals, dtype=float)

    def on_gas(self, msg: Float64):
        conc = max(float(msg.data), 0.0)
        self._last_gas_raw = conc
        self.db_hist.append(conc)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def ee_position(self):
        tf: TransformStamped = self.tf_buffer.lookup_transform(
            self.world, self.ee, rclpy.time.Time(), timeout=Duration(seconds=10.0)
        )
        p = tf.transform.translation
        return np.array([float(p.x), float(p.y), float(p.z)], dtype=float)

    def sensor_position(self):
        tf: TransformStamped = self.tf_buffer.lookup_transform(
            self.world, self.sensor_frame, rclpy.time.Time(), timeout=Duration(seconds=10.0)
        )
        p = tf.transform.translation
        return np.array([float(p.x), float(p.y), float(p.z)], dtype=float)

    def distance_to_source(self) -> float:
        try:
            sensor_pos = self.sensor_position()
            return float(np.linalg.norm(sensor_pos - self.src))
        except Exception:
            return 1e9

    # ------------------------------------------------------------------ #
    # 启动阶段检测
    # ------------------------------------------------------------------ #
    def _startup_micro_step(self):
        if self._startup_q0 is None or self._q_cmd is None:
            return
        dq = np.zeros_like(self._q_cmd)
        micro_step = min(self.max_step, math.radians(5.0))
        t_now = self.get_clock().now().nanoseconds * 1e-9

        for j, freq, phase in [(0, 0.7, 0.0),
                               (1, 0.5, 1.0),
                               (2, 0.9, 2.0),
                               (3, 0.6, 0.5),
                               (4, 0.8, 1.5)]:
            if j >= len(dq):
                break
            dq[j] = micro_step * 0.6 * math.sin(freq * t_now + phase)

        q_tgt = np.clip(self._startup_q0 + dq, self.lower, self.upper)
        self._q_cmd = q_tgt
        self._publish_cmd()

    def _classify_startup_state(self):
        if not self._startup_samples or len(self._startup_samples) < 5:
            self._startup_class = 'UNKNOWN'
            self._startup_metrics = {}
            self.get_logger().warn("[STARTUP] not enough samples, mark UNKNOWN.")
            self._startup_done = True
            return

        arr = np.array(self._startup_samples, dtype=float)
        C_mean = float(arr.mean())
        C_max = float(arr.max())
        C_min = float(arr.min())
        sigma = float(arr.std(ddof=0))

        if hasattr(self, "_startup_grad_list") and self._startup_grad_list:
            grad = float(np.mean(self._startup_grad_list))
        else:
            grad = 0.0

        background = float(self.baseline_est.baseline)
        noise = float(self.sensor_noise)

        above_background = C_mean > background + 3.0 * noise
        has_variation = sigma > 2.0 * noise
        has_peak = C_max > background + 5.0 * noise

        hit_thr = max(C_mean * 1.3, background + 4.0 * noise)
        hits = int(np.sum(arr >= hit_thr))
        hit_rate = hits / float(len(arr))

        C_abs = max(C_mean, 1.0)
        cv = sigma / C_abs

        inside_plume = above_background and (has_peak or has_variation)

        if inside_plume:
            if hit_rate >= 0.8 and cv < 0.15:
                state = 'A'
            else:
                state = 'B'
        else:
            if has_peak and hit_rate > 0.1:
                state = 'C'
            else:
                state = 'D'

        self._startup_class = state
        self._startup_metrics = {
            "C_mean": C_mean,
            "C_max": C_max,
            "C_min": C_min,
            "sigma": sigma,
            "grad": grad,
            "cv": cv,
            "hit_rate": hit_rate,
            "inside_plume": bool(inside_plume),
            "background": background,
            "noise": noise,
            "samples": int(arr.size),
        }

        self.get_logger().info(
            "[STARTUP] classify: state=%s, C_mean=%.2f, C_max=%.2f, "
            "sigma=%.3f, grad=%.3f, cv=%.3f, hit_rate=%.2f, inside=%s, n=%d"
            % (
                state,
                C_mean, C_max,
                sigma, grad, cv,
                hit_rate, str(inside_plume),
                arr.size,
            )
        )

        self._startup_done = True

    def _run_startup_detection(self):
        if not self.db_hist:
            return

        now = time.time()
        if self._startup_t0 is None:
            self._startup_t0 = now
            self._startup_phase = 'STATIC'
            if self._q_cur is not None:
                self._startup_q0 = self._q_cur.copy()
                self._q_cmd = self._startup_q0.copy()
            self._startup_grad_list = []
            self.get_logger().info("[STARTUP] begin static sampling (0–2 s).")

        t_rel = now - self._startup_t0
        cur_val = self.db_hist[-1]
        self._startup_samples.append(cur_val)

        if self._startup_last_db is not None:
            self._startup_grad_list.append(abs(cur_val - self._startup_last_db))
        self._startup_last_db = cur_val

        if t_rel < self._startup_static_s:
            self._startup_phase = 'STATIC'
            if self._startup_q0 is not None:
                self._q_cmd = self._startup_q0.copy()
                self._publish_hold()
            return

        elif t_rel < self._startup_static_s + self._startup_micro_s:
            if self._startup_phase != 'MICRO':
                self._startup_phase = 'MICRO'
                self.get_logger().info("[STARTUP] enter micro-move probing (2–4 s).")
            self._startup_micro_step()
            return

        elif t_rel < self._startup_total_s:
            if self._startup_phase != 'CLASSIFY':
                self._startup_phase = 'CLASSIFY'
                self.get_logger().info("[STARTUP] classifying initial state.")
                self._classify_startup_state()
            self._publish_hold()
            return

        else:
            if not self._startup_done:
                self.get_logger().warn("[STARTUP] timeout; force classification/end.")
                if self._startup_phase != 'CLASSIFY':
                    self._classify_startup_state()
            self._startup_done = True
            self.get_logger().info(
                f"[STARTUP] done, initial_state={self._startup_class or 'UNKNOWN'}."
            )
            return

    # ------------------------------------------------------------------ #
    # plume 阶段 / 命中统计
    # ------------------------------------------------------------------ #
    def compute_margin(self, conc: float, baseline: float) -> float:
        """
        统一 margin 计算，带baseline保护
        
        ★ 关键修复：在SURGE/SPIRAL期间，使用进入时的baseline
        防止baseline被高浓度区域拉高，导致离开时margin变成很负
        """
        # ★ 选择合适的baseline ★
        effective_baseline = baseline
        
        # SURGE期间：使用进入时的baseline（如果当前baseline更高）
        if self.state == 'SURGE' and self._surge_entry_baseline is not None:
            if baseline > self._surge_entry_baseline * 1.2:  # baseline被拉高超过20%
                effective_baseline = self._surge_entry_baseline
        
        # SPIRAL期间：使用进入时的baseline（如果当前baseline更高）
        if self.state == 'SPIRAL' and self._spiral_entry_baseline is not None:
            if baseline > self._spiral_entry_baseline * 1.2:
                effective_baseline = self._spiral_entry_baseline
        
        b = max(effective_baseline, self.baseline_min)
        if self.signal_mode == 'ratio':
            return (conc - b) / b
        else:
            return conc - b

    def _update_plume_stats(self, margin: float):
        if self.signal_mode == 'ratio':
            thr = self.margin_mid
        else:
            thr = self.delta_mid
        
        hit = margin >= thr
        self._hit_short.append(1 if hit else 0)
        self._hit_long.append(1 if hit else 0)
        self._margin_short.append(margin)
        self._margin_long.append(margin)

    def _plume_hit_rates(self):
        hit_rate_short = sum(self._hit_short) / len(self._hit_short) if self._hit_short else 0.0
        hit_rate_long = sum(self._hit_long) / len(self._hit_long) if self._hit_long else 0.0
        return hit_rate_short, hit_rate_long

    def _plume_grad_short(self):
        if len(self._margin_short) < 2:
            return 0.0
        return (self._margin_short[-1] - self._margin_short[0]) / max(1, len(self._margin_short))

    def _update_plume_phase(self, margin: float):
        if self._hit_state == 1:
            if margin <= self._plume_off_margin:
                self._hit_state = 0
        else:
            if margin >= self._plume_on_margin:
                self._hit_state = 1

        self._hit_hist_phase.append(self._hit_state)
        self._margin_hist_phase.append(float(margin))

        if len(self._hit_hist_phase) < max(3, int(0.5 * self._plume_window_steps)):
            self._prev_plume_phase = self.plume_phase
            return

        hit_ratio = float(sum(self._hit_hist_phase)) / float(len(self._hit_hist_phase))

        gap_cur = 0
        gap_max_steps = 0
        for h in self._hit_hist_phase:
            if h == 0:
                gap_cur += 1
                gap_max_steps = max(gap_max_steps, gap_cur)
            else:
                gap_cur = 0
        gap_max_s = gap_max_steps * self.dt

        margin_peak = max(self._margin_hist_phase) if self._margin_hist_phase else margin

        prev = self.plume_phase
        phase = prev

        if hit_ratio <= self._plume_bg_ratio and gap_max_s >= self._plume_bg_gap_s:
            phase = 'NO_CONTACT'
            self._plume_core_time = 0.0
        else:
            if hit_ratio >= self._plume_core_ratio and margin_peak >= self._plume_on_margin:
                self._plume_core_time += self.dt
                if self._plume_core_time >= self._plume_candidate_time_s:
                    phase = 'CONFIRMED'
                else:
                    phase = 'CANDIDATE'
            elif hit_ratio >= (self._plume_bg_ratio * 2.0) and margin_peak >= self._plume_on_margin:
                phase = 'CANDIDATE'
                self._plume_core_time = 0.0
            else:
                self._plume_core_time = 0.0
                if prev == 'CONFIRMED':
                    phase = 'CANDIDATE'
                elif prev == 'CANDIDATE':
                    phase = 'CANDIDATE' if hit_ratio > 0.0 else 'NO_CONTACT'
                else:
                    phase = 'NO_CONTACT'

        ee = getattr(self, "_ee_for_phase", None)

        if phase == 'CONFIRMED' and ee is not None:
            if self._global_core_peak_margin > -1e8:
                target_peak = self._global_core_peak_margin
            else:
                target_peak = margin_peak

            if margin >= target_peak - 0.5:
                if self._core_center_pos is None:
                    self._core_center_pos = ee.copy()
                    self._core_center_margin = margin
                else:
                    alpha = 0.1
                    self._core_center_pos = (1.0 - alpha) * self._core_center_pos + alpha * ee
                    self._core_center_margin = max(self._core_center_margin, margin)

        if phase == 'CONFIRMED' and self._core_center_pos is not None and ee is not None:
            d = float(np.linalg.norm(ee - self._core_center_pos))
            margin_drop = max(0.0, self._core_center_margin - margin)
            drop_norm = min(1.0, margin_drop / 2.0)
            r_allow = self._core_radius_max - drop_norm * (self._core_radius_max - self._core_radius_min)
            if d > r_allow:
                phase = 'CANDIDATE'

        self._prev_plume_phase = prev
        self.plume_phase = phase

    # ------------------------------------------------------------------ #
    # plume 区域分类
    # ------------------------------------------------------------------ #
    def _classify_plume_zone(self, margin: float) -> str:
        phase = self.plume_phase

        if self.signal_mode == 'ratio':
            low, mid, high = self.margin_low, self.margin_mid, self.margin_high
        else:
            low, mid, high = self.delta_low, self.delta_mid, self.delta_high

        hit_rate_short, _ = self._plume_hit_rates()

        if margin >= high:
            base_zone = 'core'
        elif margin >= mid:
            base_zone = 'edge'
        elif margin >= low:
            base_zone = 'near'
        else:
            base_zone = 'far'

        if base_zone == 'core':
            if hit_rate_short < 0.2:
                zone = 'edge'
            else:
                zone = 'core'
        
        elif base_zone == 'edge':
            if hit_rate_short > 0.5:
                zone = 'core'
            elif hit_rate_short < 0.1:
                zone = 'near'
            else:
                zone = 'edge'
        
        elif base_zone == 'near':
            if hit_rate_short > 0.35:
                zone = 'edge'
            elif hit_rate_short < 0.02:
                zone = 'far'
            else:
                zone = 'near'
        
        else:
            if hit_rate_short > 0.05:
                zone = 'near'
            else:
                zone = 'far'

        if phase == 'CONFIRMED' and zone == 'far':
            zone = 'near'
        
        if phase == 'NO_CONTACT' and zone == 'core':
            zone = 'edge'

        return zone

    # ------------------------------------------------------------------ #
    # 源点判定
    # ------------------------------------------------------------------ #
    def _update_source_detector(self, margin: float, ee_now):
        prev_state = self.source_state

        if self._src_initial_ee is None and ee_now is not None:
            self._src_initial_ee = ee_now.copy()

        if self.signal_mode == 'ratio':
            hit_thr = self.margin_high
        else:
            hit_thr = self.delta_high

        hit = margin >= hit_thr
        self._src_margin_hist.append(float(margin))
        self._src_hit_hist.append(1 if hit else 0)

        if margin > self._src_peak_margin:
            self._src_peak_margin = float(margin)
            if ee_now is not None:
                self._src_peak_margin_pos = ee_now.copy()

        min_len = max(5, int(0.3 * self._src_window_steps))
        if len(self._src_margin_hist) < min_len:
            self.source_conf *= 0.98
            self.source_conf = clamp(self.source_conf, 0.0, 1.0)
            return

        arr = np.array(self._src_margin_hist, dtype=float)
        mu = float(arr.mean())
        sigma = float(arr.std()) if arr.size > 1 else 0.0
        hit_rate = float(sum(self._src_hit_hist)) / float(len(self._src_hit_hist))
        cv = sigma / (abs(mu) + 1e-3)

        n = len(arr)
        k = max(1, n // 3)
        mu_first = float(arr[:k].mean())
        mu_last = float(arr[-k:].mean())
        trend = mu_last - mu_first

        if self.signal_mode == 'ratio':
            MU_HIGH = hit_thr * 0.9
            TREND_MAX = 0.06
        else:
            MU_HIGH = hit_thr * 0.55
            TREND_MAX = 50.0

        CV_MAX = 0.60
        HIT_RATE_THR = 0.5

        score = 0.0

        if mu >= MU_HIGH:
            score += 0.35
        elif mu >= MU_HIGH * 0.7:
            score += 0.2

        if hit_rate >= HIT_RATE_THR:
            score += 0.25
        elif hit_rate >= 0.5:
            score += 0.15

        if cv <= CV_MAX:
            score += 0.2
        elif cv <= 0.5:
            score += 0.1

        if abs(trend) <= TREND_MAX:
            score += 0.15
        elif abs(trend) <= TREND_MAX * 2:
            score += 0.05

        if self._src_peak_margin > 0 and mu >= self._src_peak_margin * 0.85:
            score += 0.05

        if self.plume_phase == 'NO_CONTACT':
            score *= 0.25
        elif self.plume_phase == 'CANDIDATE':
            score *= 0.8
        else:
            score *= 1.0

        alpha = 0.08
        self.source_conf = (1.0 - alpha) * self.source_conf + alpha * score

        if self.signal_mode == 'ratio':
            low = self.margin_low
        else:
            low = self.delta_low

        _, hit_rate_long = self._plume_hit_rates()
        if (self.plume_phase == 'NO_CONTACT'
                and margin < low
                and hit_rate_long < 0.1):
            self.source_conf *= 0.92

        self.source_conf = clamp(self.source_conf, 0.0, 1.0)

        t_elapsed = time.time() - self.t0
        move_dist = 0.0
        if self._src_initial_ee is not None and ee_now is not None:
            move_dist = float(np.linalg.norm(ee_now - self._src_initial_ee))

        now_t = time.time()

        if self.source_state == 'NONE':
            if self.source_conf >= 0.6 and self.plume_phase in ('CANDIDATE', 'CONFIRMED'):
                self.source_state = 'CANDIDATE'
                self._src_candidate_t0 = now_t
                self._src_confirm_t0 = None

        elif self.source_state == 'CANDIDATE':
            can_confirm = True

            if self.source_conf < 0.75:
                can_confirm = False

            if self.plume_phase not in ('CANDIDATE', 'CONFIRMED'):
                can_confirm = False

            if t_elapsed < self._src_min_run_time_s:
                can_confirm = False

            if move_dist < self._src_min_move_dist:
                can_confirm = False

            if abs(trend) > TREND_MAX * 1.5:
                can_confirm = False

            if can_confirm:
                if self._src_confirm_t0 is None:
                    self._src_confirm_t0 = now_t
                elif now_t - self._src_confirm_t0 >= 2.0:
                    self.source_state = 'CONFIRMED'
            else:
                self._src_confirm_t0 = None

            if self.source_conf <= 0.2 and self.plume_phase == 'NO_CONTACT':
                self.source_state = 'NONE'
                self._src_candidate_t0 = None
                self._src_confirm_t0 = None

        elif self.source_state == 'CONFIRMED':
            if (self.source_conf <= 0.35
                    and self.plume_phase == 'NO_CONTACT'
                    and hit_rate_long < 0.1):
                self.source_state = 'NONE'
                self._src_candidate_t0 = None
                self._src_confirm_t0 = None

        if prev_state != 'CONFIRMED' and self.source_state == 'CONFIRMED':
            self.get_logger().info(
                "[SOURCE] FOUND - state=CONFIRMED, "
                f"mu={mu:.2f}, hit_rate={hit_rate:.2f}, cv={cv:.3f}, trend={trend:+.3f}, "
                f"t_elapsed={t_elapsed:.1f}s, move_dist={move_dist:.3f}m"
            )

    # ------------------------------------------------------------------ #
    # SURGE 控制
    # ------------------------------------------------------------------ #
    def _surge_by_signal(self, margin: float):
        dq = np.zeros_like(self._q_cmd)
        cur_gas = self.db_hist[-1] if self.db_hist else self.baseline
        zone = getattr(self, 'plume_zone', 'near')
        t_now_s = time.time()

        cast_weight = self._direction_estimate.get_weight(t_now_s, self.cast_dir_decay_time_s)
        local_weight = 1.0 - cast_weight
        
        cast_yaw = self._direction_estimate.yaw_dir if self._direction_estimate.valid else 0.0
        cast_reach = self._direction_estimate.reach_dir if self._direction_estimate.valid else 0.0
        cast_pitch = self._direction_estimate.pitch_dir if self._direction_estimate.valid else 0.0

        local_yaw_grad = 0.0
        if len(self._side_left) >= 2 and len(self._side_right) >= 2:
            ml = sum(self._side_left) / len(self._side_left)
            mr = sum(self._side_right) / len(self._side_right)
            if abs(mr - ml) > self.signal_side_eps:
                local_yaw_grad = np.sign(mr - ml)

        local_reach_grad = 0.0
        if len(self._side_forward) >= 2 and len(self._side_backward) >= 2:
            mf = sum(self._side_forward) / len(self._side_forward)
            mb = sum(self._side_backward) / len(self._side_backward)
            if abs(mf - mb) > self.signal_side_eps:
                local_reach_grad = np.sign(mf - mb)

        local_pitch_grad = 0.0
        if len(self._side_up) >= 2 and len(self._side_down) >= 2:
            mu = sum(self._side_up) / len(self._side_up)
            md = sum(self._side_down) / len(self._side_down)
            if abs(mu - md) > self.signal_side_eps:
                local_pitch_grad = np.sign(mu - md)

        yaw_dir = cast_weight * cast_yaw + local_weight * local_yaw_grad
        reach_dir = cast_weight * cast_reach + local_weight * local_reach_grad
        pitch_dir = cast_weight * cast_pitch + local_weight * local_pitch_grad

        grad_short = self._plume_grad_short()
        
        # ★ 修复：如果CAST方向完全过期，使用保存的方向 ★
        if cast_weight < 0.05:
            # 使用_last_good_*作为备用（可能来自SPIRAL）
            if abs(self._last_good_reach_dir) > 0.1:
                reach_dir = 0.5 * reach_dir + 0.5 * self._last_good_reach_dir
            if abs(self._last_good_yaw_dir) > 0.1:
                yaw_dir = 0.5 * yaw_dir + 0.5 * self._last_good_yaw_dir
            if abs(self._last_good_pitch_dir) > 0.1:
                pitch_dir = 0.5 * pitch_dir + 0.5 * self._last_good_pitch_dir
        
        if cast_weight < 0.15:
            if reach_dir < 0.4:
                reach_dir = 0.8
        
        if grad_short > 2.0 and reach_dir < 0.5:
            reach_dir = max(reach_dir, 0.7)
        
        reach_dir = max(reach_dir, 0.3)

        # 记录当前方向（供SPIRAL使用）
        self._last_good_yaw_dir = yaw_dir
        self._last_good_reach_dir = reach_dir
        self._last_good_pitch_dir = pitch_dir

        if zone == 'far':
            base_reach = self.reach_step * 2.0
            yaw_step = self.yaw_step * 1.2
            pitch_step = self.pitch_step * 0.8
        elif zone == 'near':
            base_reach = self.reach_step * 1.6
            yaw_step = self.yaw_step * 1.0
            pitch_step = self.pitch_step * 0.7
        elif zone == 'edge':
            base_reach = self.reach_step * 1.4
            yaw_step = self.yaw_step * 1.0
            pitch_step = self.pitch_step * 0.8
        else:
            base_reach = self.reach_step * 1.0
            yaw_step = self.yaw_step * 0.8
            pitch_step = self.pitch_step * 0.6

        dq[0] += yaw_step * yaw_dir
        dq[1] += base_reach * reach_dir
        dq[2] += 0.5 * pitch_step * pitch_dir

        t_now = self.get_clock().now().nanoseconds * 1e-9
        if len(dq) >= 5:
            dq[4] += 0.08 * self.pitch_step * math.cos(0.4 * t_now)
        if len(dq) >= 6:
            dq[5] += 0.06 * self.pitch_step * math.sin(0.5 * t_now)

        if len(dq) >= 7:
            dq[6] += math.radians(12.0)

        if dq[0] > 1e-6:
            self._side_right.append(margin)
        elif dq[0] < -1e-6:
            self._side_left.append(margin)
        
        if dq[1] > 0.01:
            self._side_forward.append(margin)
        elif dq[1] < -0.01:
            self._side_backward.append(margin)
        
        pitch_total = dq[2] + dq[3] if len(dq) > 3 else dq[2]
        if pitch_total > 0.01:
            self._side_up.append(margin)
        elif pitch_total < -0.01:
            self._side_down.append(margin)

        return np.clip(dq, -self.max_step, self.max_step)

    # ------------------------------------------------------------------ #
    # CAST 扫动
    # ------------------------------------------------------------------ #
    def _cast_sweep(self, margin: float):
        dq = np.zeros_like(self._q_cmd)
        if self._q_cmd is None:
            return dq

        t_now = self.get_clock().now().nanoseconds * 1e-9
        zone = getattr(self, 'plume_zone', 'near')

        if self._cast_start_time is None:
            self._reset_cast_sampling()

        if zone == 'far':
            A_h, A_v, A_r = math.radians(40), math.radians(35), math.radians(20)
            omega = 0.8
        elif zone == 'near':
            A_h, A_v, A_r = math.radians(25), math.radians(22), math.radians(15)
            omega = 1.0
        elif zone == 'edge':
            A_h, A_v, A_r = math.radians(35), math.radians(30), math.radians(20)
            omega = 0.9
        else:
            A_h, A_v, A_r = math.radians(10), math.radians(10), math.radians(5)
            omega = 1.2

        if self.plume_phase == 'CONFIRMED':
            A_h *= 0.5
            A_v *= 0.6
            A_r *= 0.5
            omega *= 1.3
        elif self.plume_phase == 'CANDIDATE':
            A_h *= 0.7
            A_v *= 0.8
            A_r *= 0.7

        t_away_s = self._cast_steps_since_core * self.dt
        amp_scale = 1.0 + min(1.5, t_away_s / 4.0)
        A_h *= amp_scale
        A_v *= amp_scale
        A_r *= amp_scale

        lr_bias = 0.0
        if len(self._cast_left) >= 3 and len(self._cast_right) >= 3:
            ml = sum(self._cast_left) / len(self._cast_left)
            mr = sum(self._cast_right) / len(self._cast_right)
            if abs(mr - ml) > self.cast_side_eps:
                lr_bias = 0.3 * np.sign(mr - ml)

        fb_bias = 0.0
        if len(self._side_forward) >= 3 and len(self._side_backward) >= 3:
            mf = sum(self._side_forward) / len(self._side_forward)
            mb = sum(self._side_backward) / len(self._side_backward)
            if abs(mf - mb) > self.signal_side_eps:
                fb_bias = 0.3 * np.sign(mf - mb)

        ud_bias = 0.0
        if len(self._side_up) >= 3 and len(self._side_down) >= 3:
            mu = sum(self._side_up) / len(self._side_up)
            md = sum(self._side_down) / len(self._side_down)
            if abs(mu - md) > self.signal_side_eps:
                ud_bias = 0.3 * np.sign(mu - md)

        theta = A_h * math.sin(omega * t_now) + lr_bias * A_h
        phi = A_v * math.cos(omega * t_now + math.pi / 4) + fb_bias * A_v
        psi = A_r * math.sin(2 * omega * t_now) + ud_bias * A_r

        xi = -math.pi / 2 + 0.2 * math.sin(omega * t_now + math.pi / 2) + ud_bias * 0.1
        rho = 0.3 * math.sin(1.5 * omega * t_now)
        tau = math.pi / 4 + 0.1 * math.sin(omega * t_now)

        gamma_dot = math.radians(20.0)

        q = self._q_cmd.copy()
        target = np.array([
            theta,
            phi,
            psi,
            xi,
            rho,
            tau,
            q[6]
        ])

        k_traj = 0.28
        dq = k_traj * (target - q)

        if len(dq) >= 7:
            dq[6] += gamma_dot

        cast_probe = self.cast_probe_yaw_step * (1 if self._cast_probe_sign >= 0 else -1)
        self._cast_probe_sign = -self._cast_probe_sign if self._cast_probe_sign != 0 else 1
        dq[0] += 0.3 * cast_probe

        if cast_probe > 0:
            self._cast_right.append(margin)
            self._cast_samples_right.append(margin)
        else:
            self._cast_left.append(margin)
            self._cast_samples_left.append(margin)

        if dq[0] > 1e-6:
            self._side_right.append(margin)
        elif dq[0] < -1e-6:
            self._side_left.append(margin)

        if dq[1] > 0.01:
            self._side_forward.append(margin)
            self._cast_samples_forward.append(margin)
        elif dq[1] < -0.01:
            self._side_backward.append(margin)
            self._cast_samples_backward.append(margin)

        pitch_total = dq[2] + dq[3]
        if pitch_total > 0.01:
            self._side_up.append(margin)
            self._cast_samples_up.append(margin)
        elif pitch_total < -0.01:
            self._side_down.append(margin)
            self._cast_samples_down.append(margin)

        if not self._cast_ready_for_surge:
            if self._cast_samples_sufficient():
                self._compute_direction_estimate()
                de = self._direction_estimate
                
                recon_done = (
                    de.confidence >= self.cast_dir_confidence_threshold or
                    (de.reach_conf >= 0.25 and abs(de.reach_dir) > 0.25)
                )
                
                if recon_done:
                    self._cast_ready_for_surge = True
                    self.get_logger().info(
                        f'[CAST] 侦察完成！准备切换到 SURGE，方向: '
                        f'reach={de.reach_dir:.2f}, yaw={de.yaw_dir:.2f}, '
                        f'conf={de.confidence:.2f}, reach_c={de.reach_conf:.2f}'
                    )

        drift_base = {
            'far': 0.5, 'near': 0.4, 'edge': 0.3, 'core': 0.15
        }.get(zone, 0.4)

        if len(self._cast_left) >= 3 and len(self._cast_right) >= 3:
            ml = sum(self._cast_left) / len(self._cast_left)
            mr = sum(self._cast_right) / len(self._cast_right)

            if abs(mr - ml) > self.cast_side_eps:
                dq[0] += np.sign(mr - ml) * drift_base * self.yaw_step

        if len(self._side_forward) >= 3 and len(self._side_backward) >= 3:
            mf = sum(self._side_forward) / len(self._side_forward)
            mb = sum(self._side_backward) / len(self._side_backward)

            if abs(mf - mb) > self.signal_side_eps:
                dq[1] += np.sign(mf - mb) * drift_base * self.reach_step

        return np.clip(dq, -self.max_step, self.max_step)

    # ------------------------------------------------------------------ #
    # CAST 侦察兵方法
    # ------------------------------------------------------------------ #
    def _reset_cast_sampling(self):
        self._cast_start_time = time.time()
        self._cast_ready_for_surge = False
        self._cast_samples_left = []
        self._cast_samples_right = []
        self._cast_samples_forward = []
        self._cast_samples_backward = []
        self._cast_samples_up = []
        self._cast_samples_down = []

    def _cast_samples_sufficient(self) -> bool:
        n_lr = min(len(self._cast_samples_left), len(self._cast_samples_right))
        n_fb = min(len(self._cast_samples_forward), len(self._cast_samples_backward))
        
        if n_lr < self.cast_min_samples:
            return False
        
        if self._cast_start_time is None:
            return False
        cast_duration = time.time() - self._cast_start_time
        if cast_duration < self.cast_min_duration_s:
            return False
        
        def cv(arr):
            if len(arr) < 3:
                return 0.0
            m = np.mean(arr)
            if m < 1e-6:
                return 0.0
            return np.std(arr) / m
        
        if cv(self._cast_samples_left) > self.cast_cv_max:
            return False
        if cv(self._cast_samples_right) > self.cast_cv_max:
            return False
        
        return True

    def _compute_direction_estimate(self):
        if self.state == 'SURGE':
            return
        
        de = self._direction_estimate
        
        if self._cast_samples_left and self._cast_samples_right:
            ml = np.mean(self._cast_samples_left)
            mr = np.mean(self._cast_samples_right)
            diff_lr = mr - ml
            de.gradient_lr = diff_lr
            
            if abs(diff_lr) > 1.0:
                de.yaw_dir = np.clip(diff_lr / 5.0, -1.0, 1.0)
            else:
                de.yaw_dir = 0.0
            
            de.yaw_conf = min(1.0, abs(diff_lr) / 10.0)
            de.n_left = len(self._cast_samples_left)
            de.n_right = len(self._cast_samples_right)
        else:
            de.yaw_dir = 0.0
            de.yaw_conf = 0.0
            de.gradient_lr = 0.0
        
        grad_fb = self._plume_grad_short()
        de.gradient_fb = grad_fb
        
        if abs(grad_fb) > 1.5:
            de.reach_dir = np.clip(grad_fb / 5.0, -1.0, 1.0)
        else:
            de.reach_dir = 0.0
        
        de.reach_conf = min(1.0, abs(grad_fb) / 10.0)
        de.n_forward = len(self._cast_samples_forward) if self._cast_samples_forward else 0
        de.n_backward = len(self._cast_samples_backward) if self._cast_samples_backward else 0
        
        if self._cast_samples_up and self._cast_samples_down:
            mu = np.mean(self._cast_samples_up)
            md = np.mean(self._cast_samples_down)
            diff_ud = mu - md
            de.gradient_ud = diff_ud
            
            if abs(diff_ud) > 1.0:
                de.pitch_dir = np.clip(diff_ud / 5.0, -1.0, 1.0)
            else:
                de.pitch_dir = 0.0
            
            de.pitch_conf = min(1.0, abs(diff_ud) / 10.0)
            de.n_up = len(self._cast_samples_up)
            de.n_down = len(self._cast_samples_down)
        else:
            de.pitch_dir = 0.0
            de.pitch_conf = 0.0
            de.gradient_ud = 0.0
        
        de.confidence = 0.5 * de.yaw_conf + 0.35 * de.reach_conf + 0.15 * de.pitch_conf
        de.timestamp = time.time()
        
        if de.reach_conf >= 0.25 and abs(de.reach_dir) > 0.25:
            de.valid = True
        else:
            de.valid = (de.confidence > 0.12)
        
        self.get_logger().info(
            f'[CAST] 方向估计: reach={de.reach_dir:+.2f}(c={de.reach_conf:.2f}, grad={de.gradient_fb:+.1f}), '
            f'yaw={de.yaw_dir:+.2f}(c={de.yaw_conf:.2f}, g={de.gradient_lr:+.1f}), '
            f'pitch={de.pitch_dir:+.2f}(c={de.pitch_conf:.2f}), '
            f'总置信={de.confidence:.2f}, valid={de.valid}, '
            f'samples=[L{de.n_left}/R{de.n_right}]'
        )

    # ------------------------------------------------------------------ #
    # SPIRAL v9: DRIFT + PROBE 循环
    # ------------------------------------------------------------------ #
    def _init_spiral(self, margin: float):
        """初始化SPIRAL状态"""
        self._spiral_q0 = self._q_cmd.copy()
        self._spiral_t0 = time.time()
        self._spiral_substate = 'DRIFT'
        self._spiral_substate_t0 = time.time()
        
        # 初始化推进方向（来自SURGE）
        self._spiral_drift_dir = np.array([
            self._last_good_yaw_dir * 0.3,
            max(self._last_good_reach_dir, 0.5),
            self._last_good_pitch_dir * 0.2
        ])
        
        # 重置衰减系数
        self._spiral_drift_scale = 1.0
        self._spiral_probe_scale = 1.0
        
        # 清空采样
        self._spiral_probe_samples = []
        self._spiral_probe_best_offset = np.zeros(2)
        
        # 清空历史
        self._spiral_best_margin_hist.clear()
        self._spiral_gradient_hist.clear()
        
        # 重置计数
        self._spiral_fail_count = 0
        self._spiral_no_improve_count = 0
        
        # 记录最佳
        self._spiral_best_margin = margin
        self._spiral_best_q = self._q_cmd.copy()
        
        self.get_logger().info(
            f'[SPIRAL] ★初始化★ drift_dir={self._spiral_drift_dir}, '
            f'margin={margin:.1f}'
        )

    def _switch_spiral_substate(self, new_substate: str):
        """切换SPIRAL子状态"""
        old = self._spiral_substate
        self._spiral_substate = new_substate
        self._spiral_substate_t0 = time.time()
        self.get_logger().info(f'[SPIRAL] 子状态切换: {old} -> {new_substate}')

    def _spiral_local(self, margin: float):
        """
        SPIRAL v9: DRIFT + PROBE 循环
        
        - DRIFT: 小步推进 + 螺旋摆动（类似微型SURGE）
        - PROBE: 局部微扰采样 + 梯度估计（类似微型CAST）
        - 随时间衰减，最终收敛到精细定位
        """
        dq = np.zeros_like(self._q_cmd)
        if self._q_cmd is None:
            return dq
        
        t_now = time.time()
        
        # ========== 初始化 ==========
        if self._spiral_q0 is None or self._spiral_t0 is None:
            self._init_spiral(margin)
        
        # ========== 更新最佳记录 ==========
        if margin > self._spiral_best_margin:
            self._spiral_best_margin = margin
            self._spiral_best_q = self._q_cmd.copy()
            self._spiral_no_improve_count = 0
        else:
            self._spiral_no_improve_count += 1
        
        # ========== 收敛检测 ==========
        if self._check_spiral_converged(margin):
            if self._spiral_substate != 'HOLD_FINAL':
                self._switch_spiral_substate('HOLD_FINAL')
        
        # ========== 子状态机 ==========
        if self._spiral_substate == 'DRIFT':
            dq = self._spiral_drift(margin)
            # 切换到PROBE
            if t_now - self._spiral_substate_t0 >= self.spiral_drift_duration:
                self._switch_spiral_substate('PROBE')
                self._spiral_probe_samples = []  # 清空采样
        
        elif self._spiral_substate == 'PROBE':
            dq = self._spiral_probe(margin)
            # 切换到DRIFT
            if t_now - self._spiral_substate_t0 >= self.spiral_probe_duration:
                self._finalize_probe()  # 计算梯度，更新drift方向
                self._apply_spiral_decay()  # 衰减步长/振幅
                self._spiral_q0 = self._q_cmd.copy()  # 更新中心点
                self._switch_spiral_substate('DRIFT')
        
        elif self._spiral_substate == 'HOLD_FINAL':
            dq = self._spiral_hold()
        
        return np.clip(dq, -self.max_step, self.max_step)

    def _spiral_drift(self, margin: float):
        """
        SPIRAL推进模式：小步前进 + 螺旋摆动
        """
        dq = np.zeros_like(self._q_cmd)
        t = time.time() - self._spiral_t0
        
        # ========== 1. 主推进方向 ==========
        drift_dir = self._spiral_drift_dir
        
        # 基础推进步长（比SURGE小）
        base_step = self.reach_step * 0.5 * self._spiral_drift_scale
        
        # ========== 2. 螺旋摆动叠加 ==========
        omega = 1.2 * (1.0 + 0.5 * (1.0 - self._spiral_drift_scale))  # 衰减时加快频率
        
        # 左右摆动
        yaw_amp = self.yaw_step * 0.35 * self._spiral_drift_scale
        yaw_osc = math.sin(omega * t) * yaw_amp
        
        # 上下摆动（相位差90度形成螺旋）
        pitch_amp = self.pitch_step * 0.3 * self._spiral_drift_scale
        pitch_osc = math.cos(omega * t) * pitch_amp
        
        # ========== 3. 组合运动 ==========
        # J1: 左右（drift方向 + 摆动）
        dq[0] += drift_dir[0] * base_step + yaw_osc
        
        # J2: 前后（主推进）
        dq[1] += drift_dir[1] * base_step
        
        # J3: 上下（drift方向 + 摆动）
        dq[2] += drift_dir[2] * base_step * 0.5 + pitch_osc * 0.5
        
        # J4: 配合上下
        if len(dq) > 3:
            dq[3] += pitch_osc * 0.3
        
        # J5/J6: 配合摆动
        if len(dq) > 5:
            dq[4] += yaw_osc * 0.4
            dq[5] += pitch_osc * 0.4
        
        # J7: 持续滚转
        if len(dq) > 6:
            dq[6] += math.radians(10.0)
        
        return dq

    def _spiral_probe(self, margin: float):
        """
        SPIRAL采样模式：局部微扰 + 梯度估计
        """
        dq = np.zeros_like(self._q_cmd)
        dt = time.time() - self._spiral_substate_t0
        
        # ========== Lissajous微扰 ==========
        # 振幅随衰减系数减小
        amp_j5 = math.radians(15.0) * self._spiral_probe_scale
        amp_j6 = math.radians(18.0) * self._spiral_probe_scale
        
        omega_j5 = 1.2
        omega_j6 = 1.8  # 不同频率形成2D覆盖
        
        offset_j5 = amp_j5 * math.sin(omega_j5 * dt)
        offset_j6 = amp_j6 * math.sin(omega_j6 * dt)
        
        # 自适应范围（考虑关节限位）
        q0 = self._spiral_q0
        
        # J5
        margin_j5_up = self.upper[4] - q0[4]
        margin_j5_down = q0[4] - self.lower[4]
        if offset_j5 > 0:
            offset_j5 = min(offset_j5, margin_j5_up * 0.8)
        else:
            offset_j5 = max(offset_j5, -margin_j5_down * 0.8)
        
        # J6
        margin_j6_up = self.upper[5] - q0[5]
        margin_j6_down = q0[5] - self.lower[5]
        if offset_j6 > 0:
            offset_j6 = min(offset_j6, margin_j6_up * 0.8)
        else:
            offset_j6 = max(offset_j6, -margin_j6_down * 0.8)
        
        # 目标位置
        target_j5 = q0[4] + offset_j5
        target_j6 = q0[5] + offset_j6
        
        # 增量控制
        if len(dq) > 5:
            dq[4] = 0.65 * (target_j5 - self._q_cmd[4])
            dq[5] = 0.65 * (target_j6 - self._q_cmd[5])
        
        # J7: 小幅滚转
        if len(dq) > 6:
            dq[6] += math.radians(5.0)
        
        # ========== 记录采样 ==========
        self._spiral_probe_samples.append({
            'offset_j5': offset_j5,
            'offset_j6': offset_j6,
            'margin': margin,
            't': dt
        })
        
        return dq

    def _finalize_probe(self):
        """PROBE结束：计算梯度，更新drift方向"""
        if not self._spiral_probe_samples:
            return
        
        samples = self._spiral_probe_samples
        
        # 找最佳采样点
        best = max(samples, key=lambda x: x['margin'])
        self._spiral_probe_best_offset = np.array([best['offset_j5'], best['offset_j6']])
        best_margin = best['margin']
        
        # 记录best_margin历史
        self._spiral_best_margin_hist.append(best_margin)
        
        # ========== 估计局部梯度 ==========
        # 分组：J5左vs右，J6上vs下
        j5_left = [s['margin'] for s in samples if s['offset_j5'] < -0.02]
        j5_right = [s['margin'] for s in samples if s['offset_j5'] > 0.02]
        j6_down = [s['margin'] for s in samples if s['offset_j6'] < -0.02]
        j6_up = [s['margin'] for s in samples if s['offset_j6'] > 0.02]
        
        grad_lr = 0.0
        if j5_left and j5_right:
            grad_lr = np.mean(j5_right) - np.mean(j5_left)
        
        grad_ud = 0.0
        if j6_down and j6_up:
            grad_ud = np.mean(j6_up) - np.mean(j6_down)
        
        # 记录梯度方向
        self._spiral_gradient_hist.append((grad_lr, grad_ud))
        
        # ========== 更新drift方向（修复：增加衰减防止累积）==========
        update_gain = 0.20 * self._spiral_drift_scale  # 降低更新增益
        
        # ★ 修复1：先衰减旧值，再加新值（防止无限累积）
        decay = 0.7  # 每轮衰减30%
        
        if abs(grad_lr) > 1.5:
            # 衰减 + 更新
            self._spiral_drift_dir[0] = decay * self._spiral_drift_dir[0] + update_gain * np.sign(grad_lr)
        else:
            # 梯度不明显时，向0衰减
            self._spiral_drift_dir[0] *= decay
        
        if abs(grad_ud) > 1.5:
            self._spiral_drift_dir[2] = decay * self._spiral_drift_dir[2] + update_gain * np.sign(grad_ud)
        else:
            self._spiral_drift_dir[2] *= decay
        
        # ★ 修复2：更严格的上下限（pitch不应该太大）
        self._spiral_drift_dir[0] = np.clip(self._spiral_drift_dir[0], -0.6, 0.6)  # yaw限制
        self._spiral_drift_dir[2] = np.clip(self._spiral_drift_dir[2], -0.4, 0.4)  # pitch限制更严
        
        # 保证有前进分量
        self._spiral_drift_dir[1] = max(self._spiral_drift_dir[1], 0.4)
        
        # ★ 修复3：同步更新_last_good_*（供SURGE使用）
        self._last_good_yaw_dir = self._spiral_drift_dir[0]
        self._last_good_reach_dir = self._spiral_drift_dir[1]
        self._last_good_pitch_dir = self._spiral_drift_dir[2]
        
        self.get_logger().info(
            f'[SPIRAL PROBE] best_margin={best_margin:.1f}, '
            f'grad=[LR={grad_lr:+.1f}, UD={grad_ud:+.1f}], '
            f'drift_dir=[{self._spiral_drift_dir[0]:+.2f}, '
            f'{self._spiral_drift_dir[1]:+.2f}, {self._spiral_drift_dir[2]:+.2f}], '
            f'scale={self._spiral_drift_scale:.2f}'
        )

    def _apply_spiral_decay(self):
        """每轮PROBE后衰减步长/振幅"""
        self._spiral_drift_scale *= self.spiral_decay_rate
        self._spiral_probe_scale *= self.spiral_decay_rate
        
        # 下限保护
        self._spiral_drift_scale = max(self._spiral_drift_scale, self.spiral_min_scale)
        self._spiral_probe_scale = max(self._spiral_probe_scale, self.spiral_min_scale)

    def _check_spiral_converged(self, margin: float) -> bool:
        """检测是否收敛完成"""
        # 条件1：振幅已衰减到最小
        scale_converged = (
            self._spiral_drift_scale <= self.spiral_min_scale * 1.1 and
            self._spiral_probe_scale <= self.spiral_min_scale * 1.1
        )
        
        # 条件2：梯度接近0（连续多轮）
        grad_converged = False
        if len(self._spiral_gradient_hist) >= 3:
            recent_grads = list(self._spiral_gradient_hist)[-3:]
            all_small = all(
                abs(g[0]) < self.spiral_converge_grad_thr and 
                abs(g[1]) < self.spiral_converge_grad_thr 
                for g in recent_grads
            )
            grad_converged = all_small
        
        # 条件3：best_margin不再提升
        margin_stable = False
        if len(self._spiral_best_margin_hist) >= 3:
            hist = list(self._spiral_best_margin_hist)[-3:]
            improvement = max(hist) - min(hist)
            margin_stable = improvement < self.spiral_converge_margin_thr
        
        # 综合判断
        if scale_converged and (grad_converged or margin_stable):
            self.get_logger().info(
                f'[SPIRAL] ★收敛检测★ scale={self._spiral_drift_scale:.2f}, '
                f'grad_conv={grad_converged}, margin_stable={margin_stable}'
            )
            return True
        
        return False

    def _spiral_hold(self):
        """收敛后的保持状态"""
        dq = np.zeros_like(self._q_cmd)
        t = time.time()
        
        # 极小幅度摆动，保持传感器活性
        if len(dq) > 5:
            dq[5] = math.radians(3.0) * math.sin(0.6 * t)
        if len(dq) > 6:
            dq[6] = math.radians(4.0)
        
        return dq

    def _reset_spiral(self):
        """重置SPIRAL状态"""
        self._spiral_q0 = None
        self._spiral_t0 = None
        self._spiral_substate = 'DRIFT'
        self._spiral_substate_t0 = None
        self._spiral_drift_scale = 1.0
        self._spiral_probe_scale = 1.0
        self._spiral_probe_samples = []
        self._spiral_best_margin_hist.clear()
        self._spiral_gradient_hist.clear()
        self._spiral_fail_count = 0
        self._spiral_no_improve_count = 0
        self._spiral_enter_count = 0

    def _save_spiral_direction(self):
        """
        保存SPIRAL期间的方向信息，供后续SURGE使用
        
        ★ 关键：SPIRAL退出时调用，确保方向信息不丢失
        """
        if hasattr(self, '_spiral_drift_dir') and self._spiral_drift_dir is not None:
            self._last_good_yaw_dir = float(self._spiral_drift_dir[0])
            self._last_good_reach_dir = float(self._spiral_drift_dir[1])
            self._last_good_pitch_dir = float(self._spiral_drift_dir[2])
            
            # 同时更新方向估计的时间戳，使其在SURGE中有效
            self._direction_estimate.yaw_dir = self._spiral_drift_dir[0]
            self._direction_estimate.reach_dir = self._spiral_drift_dir[1]
            self._direction_estimate.pitch_dir = self._spiral_drift_dir[2]
            self._direction_estimate.timestamp = time.time()
            self._direction_estimate.valid = True
            self._direction_estimate.confidence = 0.5  # 中等置信度
            
            self.get_logger().info(
                f'[SPIRAL] 保存方向: yaw={self._last_good_yaw_dir:.2f}, '
                f'reach={self._last_good_reach_dir:.2f}, pitch={self._last_good_pitch_dir:.2f}'
            )

    # ------------------------------------------------------------------ #
    # 动态关节权重
    # ------------------------------------------------------------------ #
    def _compute_joint_weights(self):
        zone = getattr(self, 'plume_zone', 'near')

        if zone == 'far':
            W_base = np.array([0.30, 0.25, 0.15, 0.10, 0.08, 0.07, 0.05])
        elif zone == 'near':
            W_base = np.array([0.25, 0.25, 0.20, 0.10, 0.08, 0.07, 0.05])
        elif zone == 'edge':
            W_base = np.array([0.20, 0.20, 0.20, 0.15, 0.10, 0.08, 0.07])
        else:
            W_base = np.array([0.10, 0.10, 0.15, 0.15, 0.20, 0.20, 0.10])

        if self._q_cmd is None:
            return W_base

        q = self._q_cmd
        span = self.upper - self.lower
        center = (self.upper + self.lower) / 2.0
        dist_to_center = np.abs(q - center) / (span / 2.0 + 1e-6)
        F_range = 1.0 - 0.7 * np.clip(dist_to_center, 0.0, 1.0)

        W = W_base * F_range
        return W

    # ------------------------------------------------------------------ #
    # SURGE 触发阈值
    # ------------------------------------------------------------------ #
    def _get_surge_thresholds(self, zone: str):
        if self.signal_mode == 'ratio':
            if zone == 'far':
                margin_thr = self.margin_low * 0.4
                hit_rate_thr = 0.02
            elif zone == 'near':
                margin_thr = self.margin_low * 0.7
                hit_rate_thr = 0.05
            elif zone == 'edge':
                margin_thr = self.margin_mid * 0.5
                hit_rate_thr = 0.10
            else:
                margin_thr = self.margin_mid * 0.7
                hit_rate_thr = 0.15
        else:
            if zone == 'far':
                margin_thr = self.delta_low * 0.3
                hit_rate_thr = 0.02
            elif zone == 'near':
                margin_thr = self.delta_low * 0.6
                hit_rate_thr = 0.05
            elif zone == 'edge':
                margin_thr = self.delta_mid * 0.4
                hit_rate_thr = 0.10
            else:
                margin_thr = self.delta_mid * 0.6
                hit_rate_thr = 0.15
        
        return margin_thr, hit_rate_thr

    def _get_surge_exit_thresholds(self, zone: str):
        if zone == 'far':
            hit_rate_exit = 0.02
            grad_exit = -15.0
        elif zone == 'near':
            hit_rate_exit = 0.05
            grad_exit = -10.0
        elif zone == 'edge':
            hit_rate_exit = 0.10
            grad_exit = -8.0
        else:
            hit_rate_exit = 0.15
            grad_exit = -5.0
        
        return hit_rate_exit, grad_exit

    # ------------------------------------------------------------------ #
    # 主循环
    # ------------------------------------------------------------------ #
    def loop(self):
        if not self._have_js or self._q_cmd is None or not self.db_hist:
            return

        if not self._startup_done:
            self._run_startup_detection()
            return

        cur_conc = self.db_hist[-1]

        self.db_hist_new.append(cur_conc)
        if len(self.db_hist_new) > self.post_escape_short_window:
            self.db_hist_new.popleft()

        self.baseline = self.baseline_est.update(self.db_hist, self.db_hist_new)
        margin = self.compute_margin(cur_conc, self.baseline)

        self._update_plume_stats(margin)

        self._gas_hist_full.append(cur_conc)
        self._base_hist_full.append(self.baseline)

        try:
            sensor_now = self.sensor_position()
        except Exception:
            sensor_now = None
        
        try:
            ee_now = self.ee_position()
        except Exception:
            ee_now = None
        
        if sensor_now is not None and self._start_ee_pos is None:
            self._start_ee_pos = sensor_now.copy()
        
        if ee_now is not None and self._prev_ee_for_travel is not None:
            self._total_travel_dist += float(np.linalg.norm(ee_now - self._prev_ee_for_travel))
        if ee_now is not None:
            self._prev_ee_for_travel = ee_now.copy()
        
        # ========== 超时检测 ==========
        elapsed_time = time.time() - self.t0
        if elapsed_time >= self.timeout_s and not self._summary_written:
            dist = float(np.linalg.norm(sensor_now - self.src)) if sensor_now is not None else float('nan')
            self.get_logger().info(
                f"[TIMEOUT] {self.timeout_s:.0f}s reached - stopping node. "
                f"dist={dist:.3f}m, margin={margin:+.2f}, "
                f"source_state={self.source_state}, source_conf={self.source_conf:.2f}"
            )
            self._write_summary(halt_type='TIMEOUT', dist=dist, margin=margin)
            self._publish_cmd()
            try:
                rclpy.shutdown()
            except Exception:
                pass
            return
        
        if ee_now is not None:
            self._ee_hist.append(ee_now.copy())
        self._margin_hist.append(float(margin))

        self._ee_for_phase = ee_now

        self._update_plume_phase(margin)

        if self.plume_phase == 'CONFIRMED' and ee_now is not None:
            if margin > self._global_core_peak_margin + 0.05:
                self._global_core_peak_margin = float(margin)
                self._global_core_peak_pos = ee_now.copy()

        self._update_source_detector(margin, ee_now)

        # ========== 追踪变量更新 ==========
        t_now = time.time()
        
        # 更新峰值source_conf
        if self.source_conf > self._peak_source_conf:
            self._peak_source_conf = self.source_conf
            self._peak_source_conf_time = t_now - self.t0
        
        # 记录首次进入CONFIRMED的时间
        if self.plume_phase == 'CONFIRMED' and self._time_first_confirmed is None:
            self._time_first_confirmed = t_now - self.t0
        
        # 更新各状态停留时间
        if self._prev_state_time is not None:
            dt_state = t_now - self._prev_state_time
            if self._prev_state_for_time == 'CAST':
                self._time_in_cast += dt_state
            elif self._prev_state_for_time == 'SURGE':
                self._time_in_surge += dt_state
            elif self._prev_state_for_time == 'SPIRAL':
                self._time_in_spiral += dt_state
        self._prev_state_time = t_now
        self._prev_state_for_time = self.state

        if self.plume_phase == 'CONFIRMED':
            self._cast_steps_since_core = 0
        else:
            self._cast_steps_since_core += 1

        self.plume_zone = self._classify_plume_zone(margin)
        zone = self.plume_zone
        
        # 记录首次进入core的时间
        if zone == 'core' and self._time_first_core is None:
            self._time_first_core = time.time() - self.t0

        dist = self.distance_to_source()
        
        # 记录初始距离
        if self._init_distance is None:
            self._init_distance = dist

        state_change = ""
        hit_rate_short, hit_rate_long = self._plume_hit_rates()
        grad_short = self._plume_grad_short()

        # ========== 优先检查：CAST → SURGE（侦察完成立即触发）==========
        if self.state == 'CAST' and self._cast_ready_for_surge and self._direction_estimate.valid:
            self.state = 'SURGE'
            self._surge_start_time = time.time()
            self._surge_entry_baseline = self.baseline  # ★ 保存进入时的baseline
            self._surge_neg_cnt = 0
            self._surge_max_margin = -1e9
            self._surge_margin_hist.clear()
            self._cast_ready_for_surge = False
            state_change = 'CAST->SURGE (侦察完成)'
            self.surge_hits = 0
            # ★ 追踪变量更新 ★
            self._num_cast_to_surge += 1
            if self._time_first_surge is None:
                self._time_first_surge = time.time() - self.t0
            self.get_logger().info(
                f'[状态切换] CAST->SURGE：侦察完成，方向估计 '
                f'reach={self._direction_estimate.reach_dir:.2f}, '
                f'yaw={self._direction_estimate.yaw_dir:.2f}, '
                f'conf={self._direction_estimate.confidence:.2f}, '
                f'baseline={self.baseline:.1f}'
            )

        # ========== SPIRAL 精细定位：只能从 SURGE 进入 ==========
        spiral_cond = (
            self.plume_phase == 'CONFIRMED' and 
            zone == 'core' and 
            self.source_conf >= 0.5 and
            self.state == 'SURGE'
        )

        if spiral_cond and self.state != 'SPIRAL':
            self._spiral_enter_count = getattr(self, '_spiral_enter_count', 0) + 1
            if self._spiral_enter_count >= 3:
                prev_state = self.state
                self.state = 'SPIRAL'
                self._spiral_entry_baseline = self.baseline  # ★ 保存进入时的baseline
                self._reset_spiral()
                state_change = f'{prev_state}->SPIRAL (精细定位)'
                # ★ 追踪变量更新 ★
                if prev_state == 'SURGE':
                    self._num_surge_to_spiral += 1
                if self._time_first_spiral is None:
                    self._time_first_spiral = time.time() - self.t0
                self.get_logger().info(
                    f'[状态切换] {state_change} | '
                    f'phase={self.plume_phase}, zone={zone}, src_conf={self.source_conf:.2f}, '
                    f'baseline={self.baseline:.1f}'
                )
        # 独立的 if：重置 spiral 计数器，不阻断后续状态机分支
        if not spiral_cond and self.state != 'SPIRAL':
            self._spiral_enter_count = 0

        # 状态机分支（独立的 if-elif 链）
        if self.state == 'SPIRAL':
            # ========== SPIRAL 退出逻辑（三类） ==========
            
            spiral_duration = time.time() - self._spiral_t0 if self._spiral_t0 else 0.0
            
            # ----- 1. 成功退出：收敛完成 -----
            if self._spiral_substate == 'HOLD_FINAL':
                hold_duration = time.time() - self._spiral_substate_t0 if self._spiral_substate_t0 else 0.0
                if hold_duration >= 2.5:
                    self.get_logger().info('[SPIRAL] 收敛完成，等待源点确认停机')
                    self.source_conf = min(1.0, self.source_conf + 0.03)
            
            # ----- 2. 失败退出：确认被破坏 -----
            fail_cond = (
                self.plume_phase == 'NO_CONTACT' or
                self.source_conf < 0.4 or
                zone == 'far'
            )
            
            if fail_cond:
                self._spiral_fail_count += 1
            else:
                self._spiral_fail_count = max(0, self._spiral_fail_count - 1)
            
            severe_fail = (self.plume_phase == 'NO_CONTACT' and margin < self.delta_low)
            
            if severe_fail or self._spiral_fail_count >= 15:
                # ★ 修复：退出前保存SPIRAL的方向信息 ★
                self._save_spiral_direction()
                
                if self.source_conf > 0.25 or margin > self.delta_low:
                    self.state = 'SURGE'
                    self._surge_start_time = time.time()
                    self._surge_entry_baseline = self._spiral_entry_baseline or self.baseline
                    self._surge_neg_cnt = 0
                    self._surge_max_margin = -1e9
                    self._surge_margin_hist.clear()
                    state_change = 'SPIRAL->SURGE (信号恢复尝试)'
                    # ★ 追踪变量更新 ★
                    self._num_spiral_to_surge += 1
                else:
                    self.state = 'CAST'
                    self._cast_start_time = time.time()  # ★ 记录进入 CAST 的时间
                    self._reset_cast_sampling()
                    state_change = 'SPIRAL->CAST (信号丢失)'
                    # ★ 追踪变量更新 ★
                    self._num_spiral_to_cast += 1
                
                self.get_logger().info(f'[状态切换] {state_change}')
                self._reset_spiral()
            
            # ----- 3. 工程退出：超时/无进展 -----
            if spiral_duration > self.spiral_max_time:
                # ★ 修复：退出前保存SPIRAL的方向信息 ★
                self._save_spiral_direction()
                
                if self.source_conf > 0.3:
                    self.state = 'SURGE'
                    self._surge_start_time = time.time()
                    self._surge_entry_baseline = self._spiral_entry_baseline or self.baseline
                    self._surge_neg_cnt = 0
                    self._surge_max_margin = -1e9
                    self._surge_margin_hist.clear()
                    # ★ 追踪变量更新 ★
                    self._num_spiral_to_surge += 1
                else:
                    self.state = 'CAST'
                    self._cast_start_time = time.time()  # ★ 记录进入 CAST 的时间
                    self._reset_cast_sampling()
                    # ★ 追踪变量更新 ★
                    self._num_spiral_to_cast += 1
                state_change = f'SPIRAL->{"SURGE" if self.source_conf > 0.3 else "CAST"} (超时)'
                self.get_logger().info(f'[状态切换] {state_change}')
                self._reset_spiral()

        elif self.state == 'CAST':
            # ========== CAST → SURGE ==========
            if self.signal_mode == 'ratio':
                margin_base = self.margin_low
            else:
                margin_base = self.delta_low
            
            # ★ 冷却期检查：防止刚进入 CAST 就立刻切换回 SURGE ★
            cast_duration = time.time() - self._cast_start_time if self._cast_start_time else 0.0
            cast_min_duration = 1.5  # 至少停留 1.5 秒采样
            in_cooldown = cast_duration < cast_min_duration
            
            cond1 = margin >= margin_base * 0.4
            cond2 = grad_short > 1.5
            cond3 = self.plume_phase in ('CONFIRMED', 'CANDIDATE')
            cond4 = hit_rate_short > 0.03
            cond5 = margin >= margin_base * 0.8
            
            conditions_met = sum([cond1, cond2, cond3, cond4])
            
            # 冷却期内不允许切换（除非信号极强）
            if in_cooldown and margin < margin_base * 1.5:
                pass  # 冷却期内只采样不切换
            elif conditions_met >= 2 or cond5:
                self.surge_hits += 1
                if self.surge_hits >= self.hits_to_enter:
                    if len(self._cast_samples_left) >= 3 and len(self._cast_samples_right) >= 3:
                        self._compute_direction_estimate()
                    
                    self.state = 'SURGE'
                    self._surge_start_time = time.time()
                    self._surge_entry_baseline = self.baseline
                    self._surge_neg_cnt = 0
                    self._surge_max_margin = -1e9
                    self._surge_margin_hist.clear()
                    self.surge_hits = 0
                    # ★ 追踪变量更新 ★
                    self._num_cast_to_surge += 1
                    if self._time_first_surge is None:
                        self._time_first_surge = time.time() - self.t0
                    
                    de = self._direction_estimate
                    triggered = []
                    if cond1: triggered.append('margin')
                    if cond2: triggered.append('梯度')
                    if cond3: triggered.append('phase')
                    if cond4: triggered.append('hit_rate')
                    if cond5: triggered.append('高margin')
                    state_change = f'CAST->SURGE ({conditions_met}条件:{",".join(triggered)})'
                    
                    self.get_logger().info(
                        f'[状态切换] {state_change} | 方向估计: '
                        f'reach={de.reach_dir:.2f}, yaw={de.yaw_dir:.2f}, '
                        f'valid={de.valid}, conf={de.confidence:.2f}, '
                        f'baseline={self.baseline:.1f}'
                    )
            else:
                self.surge_hits = max(0, self.surge_hits - 1)

        elif self.state == 'SURGE':
            # ========== SURGE → CAST ==========
            # ★ 只基于信号特征，不依赖源点距离 ★
            t_now_s = time.time()
            surge_duration = t_now_s - self._surge_start_time if self._surge_start_time else 0.0
            
            if self._surge_start_time is None:
                self._surge_start_time = t_now_s
                surge_duration = 0.0
            
            # ========== 信号特征统计 ==========
            # 记录margin历史
            if not hasattr(self, '_surge_margin_hist'):
                self._surge_margin_hist = deque(maxlen=20)
            self._surge_margin_hist.append(margin)
            
            # 记录最大margin（用于检测下降）
            if not hasattr(self, '_surge_max_margin'):
                self._surge_max_margin = margin
            else:
                self._surge_max_margin = max(self._surge_max_margin, margin)
            
            # ========== 计算信号趋势 ==========
            margin_trend = 0.0  # 正=上升，负=下降
            
            if len(self._surge_margin_hist) >= 8:
                recent = list(self._surge_margin_hist)
                n = len(recent)
                first_half = np.mean(recent[:n//2])
                second_half = np.mean(recent[n//2:])
                margin_trend = second_half - first_half
            
            # 从峰值下降的幅度
            margin_drop_from_peak = self._surge_max_margin - margin
            
            # ========== 负值计数 ==========
            margin_negative = margin < 0
            margin_very_negative = margin < -10  # 从-15改为-10
            margin_severely_negative = margin < -25  # 从-30改为-25
            
            if margin_negative:
                self._surge_neg_cnt += 1
            else:
                self._surge_neg_cnt = max(0, self._surge_neg_cnt - 1)
            
            # ========== 退出判断（只基于信号）==========
            should_exit = False
            exit_reason = ""
            
            if surge_duration < 2.0:
                # 前2秒保护期
                if margin_severely_negative:
                    should_exit = True
                    exit_reason = "严重负值(保护期)"
            else:
                # ★ 条件1：margin持续下降趋势 ★
                if margin_trend < -15 and len(self._surge_margin_hist) >= 10:
                    should_exit = True
                    exit_reason = f"margin持续下降(trend={margin_trend:.1f})"
                
                # ★ 条件2：从峰值大幅下降 ★
                elif margin_drop_from_peak > 40 and self._surge_max_margin > 60:  # 降低阈值
                    should_exit = True
                    exit_reason = f"从峰值下降(peak={self._surge_max_margin:.0f}, drop={margin_drop_from_peak:.0f})"
                
                # ★ 条件3：连续负值（降低到2次）★
                elif self._surge_neg_cnt >= 2:
                    should_exit = True
                    exit_reason = f"连续负值({self._surge_neg_cnt}次)"
                
                # 条件4：中等负值（margin < -10）
                elif margin_very_negative:
                    should_exit = True
                    exit_reason = f"中等负值({margin:.1f})"
                
                # 条件5：严重负值
                elif margin_severely_negative:
                    should_exit = True
                    exit_reason = "严重负值"
                
                # ★ 条件6：低信号且下降趋势 ★
                elif margin < self.delta_low * 0.5 and margin_trend < -5:
                    should_exit = True
                    exit_reason = f"低信号+下降(margin={margin:.1f}, trend={margin_trend:.1f})"
                
                # ★ 条件7：长时间低信号 ★
                elif surge_duration > 8.0 and margin < self.delta_low:  # 从10秒降到8秒
                    if len(self._surge_margin_hist) >= 10:
                        recent_mean = np.mean(list(self._surge_margin_hist)[-10:])
                        if recent_mean < self.delta_low:
                            should_exit = True
                            exit_reason = f"长时间低信号(mean={recent_mean:.1f})"
            
            # 调试信息 - 每一步都输出以便诊断
            self.get_logger().info(
                f'[SURGE] margin={margin:.1f}, neg={self._surge_neg_cnt}, '
                f'trend={margin_trend:+.1f}, peak={self._surge_max_margin:.0f}, '
                f'drop={margin_drop_from_peak:.0f}, should_exit={should_exit}'
            )
            
            if should_exit:
                self.get_logger().info(
                    f'[状态切换] SURGE->CAST | {exit_reason}, '
                    f'margin={margin:.1f}, dur={surge_duration:.1f}s'
                )
                self.state = 'CAST'
                self._cast_start_time = time.time()  # ★ 记录进入 CAST 的时间
                self._surge_start_time = None
                self._surge_entry_baseline = None
                self._surge_neg_cnt = 0
                self._surge_max_margin = -1e9
                self._surge_margin_hist.clear()
                self._reset_cast_sampling()
                # ★ 追踪变量更新 ★
                self._num_surge_to_cast += 1

        try:
            if self.state == 'SURGE':
                dq = self._surge_by_signal(margin)
                self._last_dq = dq.copy()
            elif self.state == 'CAST':
                dq = self._cast_sweep(margin)
            else:  # SPIRAL
                dq = self._spiral_local(margin)

            # SPIRAL状态不应用权重
            if self.state != 'SPIRAL':
                W = self._compute_joint_weights()
                dq = dq * W

            # 源点状态调制
            if self.source_state == 'CANDIDATE':
                dq *= 0.7

            elif self.source_state == 'CONFIRMED':
                t_since_lock = 0.0
                if self._src_confirm_t0 is not None:
                    t_since_lock = time.time() - self._src_confirm_t0

                if t_since_lock < self._src_hold_time:
                    dq *= 0.5
                else:
                    dq[:] = 0.0
                    if not self._source_stop_done:
                        self._source_stop_done = True
                        self.get_logger().info(
                            "[SOURCE] CONFIRMED & stable - stopping node. "
                            f"source_conf={self.source_conf:.2f}, "
                            f"phase={self.plume_phase}, zone={self.plume_zone}, "
                            f"margin={margin:+.2f}"
                        )
                        self._write_summary(halt_type='CONFIRMED', dist=dist, margin=margin)
                        self._publish_cmd()
                        try:
                            rclpy.shutdown()
                        except Exception:
                            pass
                        return

            dq = np.clip(dq, -self.max_step, self.max_step)
            q_tgt = np.clip(self._q_cmd + dq, self.lower, self.upper)
            self._q_cmd = (1.0 - self.alpha) * self._q_cmd + self.alpha * q_tgt
        except Exception as e:
            self.get_logger().warn(f'control step failed: {e}')
            return

        self._publish_cmd()
        self._log_step(margin=margin, dist=dist, state_suffix='')

        if state_change:
            self._switch_count += 1

        sec_now = int(time.time() - self.t0)
        if sec_now != self._last_print_sec:
            self._last_print_sec = sec_now
            de = self._direction_estimate
            t_now_s = time.time()
            cast_weight = de.get_weight(t_now_s, self.cast_dir_decay_time_s)
            dir_age = de.age(t_now_s) if de.timestamp > 0 else 0.0
            
            spiral_info = ""
            if self.state == 'SPIRAL':
                spiral_info = f" spiral=[{self._spiral_substate}, s={self._spiral_drift_scale:.2f}]"
            
            self.get_logger().info(
                f'[{self.state}] step={self._step_idx} margin={margin:+.2f} '
                f'dist={dist:.3f} m dir=[r={de.reach_dir:+.2f},y={de.yaw_dir:+.2f}](w={cast_weight:.2f},age={dir_age:.1f}s) '
                f'phase={self.plume_phase} zone={self.plume_zone} '
                f'src={self.source_state}:{self.source_conf:.2f}{spiral_info}'
            )

    # ------------------------------------------------------------------ #
    # 发布 / 记录
    # ------------------------------------------------------------------ #
    def _publish_cmd(self):
        jt = JointTrajectory()
        jt.joint_names = self.joint_names
        pt = JointTrajectoryPoint()
        pt.positions = self._q_cmd.tolist()
        pt.time_from_start = Duration(seconds=self.traj_duration).to_msg()
        jt.points = [pt]
        self.pub_traj.publish(jt)

    def _publish_hold(self):
        if self._q_cmd is None:
            return
        jt = JointTrajectory()
        jt.joint_names = self.joint_names
        pt = JointTrajectoryPoint()
        pt.positions = self._q_cmd.tolist()
        pt.time_from_start = Duration(seconds=self.traj_duration).to_msg()
        jt.points = [pt]
        self.pub_traj.publish(jt)

    def _log_step(self, margin: float, dist: float, state_suffix: str):
        self._step_idx += 1
        try:
            sensor_pos = self.sensor_position()
            sensor_x, sensor_y, sensor_z = float(sensor_pos[0]), float(sensor_pos[1]), float(sensor_pos[2])
        except Exception:
            sensor_pos = None
            sensor_x = sensor_y = sensor_z = float('nan')

        try:
            gas_val = float(self.db_hist[-1]) if self.db_hist else float(self.baseline)
        except Exception:
            gas_val = float('nan')

        gas_raw = float(self._last_gas_raw) if hasattr(self, "_last_gas_raw") else float('nan')
        t_now = time.time() - getattr(self, "t0", time.time())

        base_state = (self.state if not state_suffix else f"{state_suffix}-{self.state}")
        state_str = base_state

        de = self._direction_estimate
        dir_est_reach = float(de.reach_dir)
        dir_est_yaw = float(de.yaw_dir)
        dir_est_conf = float(de.confidence)

        try:
            if self.log_writer:
                self.log_writer.writerow([
                    t_now, self._step_idx,
                    gas_raw, gas_val, margin,
                    dist,
                    sensor_x, sensor_y, sensor_z,
                    float(self.src[0]) if hasattr(self, "src") else float('nan'),
                    float(self.src[1]) if hasattr(self, "src") else float('nan'),
                    float(self.src[2]) if hasattr(self, "src") else float('nan'),
                    state_str,
                    "", "", getattr(self, "_switch_count", 0),
                    dir_est_reach, dir_est_yaw, dir_est_conf,
                    self._spiral_substate if self.state == 'SPIRAL' else '',
                    self._spiral_drift_scale if self.state == 'SPIRAL' else 0.0,
                    self._spiral_probe_scale if self.state == 'SPIRAL' else 0.0
                ])
        except Exception:
            pass

        try:
            if self.log_file:
                self.log_file.flush()
        except Exception:
            pass

    def _write_summary(self, halt_type: str, dist: float, margin: float):
        if self._summary_written:
            return
        self._summary_written = True
        
        try:
            try:
                sensor_now = self.sensor_position()
                final_sensor = [float(sensor_now[0]), float(sensor_now[1]), float(sensor_now[2])]
            except Exception:
                final_sensor = [float('nan'), float('nan'), float('nan')]
            
            try:
                ee_now = self.ee_position()
                final_ee = [float(ee_now[0]), float(ee_now[1]), float(ee_now[2])]
            except Exception:
                final_ee = [float('nan'), float('nan'), float('nan')]
            
            if self._start_ee_pos is not None:
                start_sensor = [float(self._start_ee_pos[0]), float(self._start_ee_pos[1]), float(self._start_ee_pos[2])]
            else:
                start_sensor = [float('nan'), float('nan'), float('nan')]
            
            de = self._direction_estimate
            total_time = time.time() - self.t0
            
            # 计算信号统计量
            arr = np.array(self._src_margin_hist, dtype=float) if len(self._src_margin_hist) > 0 else np.array([0.0])
            final_mu = float(arr.mean()) if len(arr) > 0 else 0.0
            final_sigma = float(arr.std()) if len(arr) > 1 else 0.0
            final_cv = final_sigma / (abs(final_mu) + 1e-3)
            
            # 计算趋势
            if len(arr) >= 3:
                k = max(1, len(arr) // 3)
                mu_first = float(arr[:k].mean())
                mu_last = float(arr[-k:].mean())
                final_trend = mu_last - mu_first
            else:
                final_trend = 0.0
            
            # 计算命中率
            hit_rate_short = sum(self._hit_short) / len(self._hit_short) if self._hit_short else 0.0
            hit_rate_long = sum(self._hit_long) / len(self._hit_long) if self._hit_long else 0.0
            
            # 计算路径效率
            init_dist = self._init_distance if self._init_distance is not None else dist
            path_efficiency = init_dist / self._total_travel_dist if self._total_travel_dist > 0.01 else 0.0
            
            with open(self.summary_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["key", "value"])
                
                # ========== 实验标识 ==========
                w.writerow(["point_id", self.point_id])
                w.writerow(["trial_num", self.trial_num])
                w.writerow(["run_id", self.run_id])
                w.writerow(["start_zone", self.start_zone])
                w.writerow(["gas_state", self.gas_state])
                w.writerow(["intermittency_index", f"{self.intermittency_index:.2f}"])
                
                # ========== 结果判定 ==========
                w.writerow(["halt_type", halt_type])
                w.writerow(["success", halt_type == 'CONFIRMED'])
                
                # 计算失败阶段（基于过程的失败分类）
                if halt_type == 'TIMEOUT':
                    if self._time_first_surge is None:
                        fail_stage = "early"
                    elif self._time_first_confirmed is None:
                        fail_stage = "mid"
                    elif self._time_first_core is None:
                        fail_stage = "mid_late"
                    else:
                        fail_stage = "late"
                else:
                    fail_stage = "success"
                w.writerow(["fail_stage", fail_stage])
                
                # ========== 距离信息 ==========
                w.writerow(["init_dist_m", f"{init_dist:.4f}"])
                w.writerow(["final_dist_m", f"{dist:.4f}"])
                
                # ========== 时间信息 ==========
                w.writerow(["total_time_s", f"{total_time:.2f}"])
                w.writerow(["total_steps", self._step_idx])
                w.writerow(["total_travel_dist_m", f"{self._total_travel_dist:.4f}"])
                w.writerow(["path_efficiency", f"{path_efficiency:.4f}"])
                
                # ========== 时间节点 ==========
                w.writerow(["time_to_first_surge", f"{self._time_first_surge:.2f}" if self._time_first_surge else "nan"])
                w.writerow(["time_to_first_spiral", f"{self._time_first_spiral:.2f}" if self._time_first_spiral else "nan"])
                w.writerow(["time_to_first_confirmed", f"{self._time_first_confirmed:.2f}" if self._time_first_confirmed else "nan"])
                w.writerow(["time_to_first_core", f"{self._time_first_core:.2f}" if self._time_first_core else "nan"])
                
                # ========== 各状态停留时间 ==========
                w.writerow(["time_in_cast", f"{self._time_in_cast:.2f}"])
                w.writerow(["time_in_surge", f"{self._time_in_surge:.2f}"])
                w.writerow(["time_in_spiral", f"{self._time_in_spiral:.2f}"])
                w.writerow(["ratio_cast", f"{self._time_in_cast / total_time:.4f}" if total_time > 0 else "0"])
                w.writerow(["ratio_surge", f"{self._time_in_surge / total_time:.4f}" if total_time > 0 else "0"])
                w.writerow(["ratio_spiral", f"{self._time_in_spiral / total_time:.4f}" if total_time > 0 else "0"])
                
                # ========== 状态切换计数 ==========
                w.writerow(["switch_count", self._switch_count])
                w.writerow(["num_cast_to_surge", self._num_cast_to_surge])
                w.writerow(["num_surge_to_cast", self._num_surge_to_cast])
                w.writerow(["num_surge_to_spiral", self._num_surge_to_spiral])
                w.writerow(["num_spiral_to_surge", self._num_spiral_to_surge])
                w.writerow(["num_spiral_to_cast", self._num_spiral_to_cast])
                
                # ========== 停机时内部状态 ==========
                w.writerow(["final_state", self.state])
                w.writerow(["final_source_state", self.source_state])
                w.writerow(["final_source_conf", f"{self.source_conf:.4f}"])
                w.writerow(["final_plume_phase", self.plume_phase])
                w.writerow(["final_plume_zone", self.plume_zone])
                w.writerow(["final_margin", f"{margin:.4f}"])
                
                # ========== 停机时信号统计量 ==========
                w.writerow(["final_mu", f"{final_mu:.4f}"])
                w.writerow(["final_sigma", f"{final_sigma:.4f}"])
                w.writerow(["final_cv", f"{final_cv:.4f}"])
                w.writerow(["final_trend", f"{final_trend:.4f}"])
                w.writerow(["final_hit_rate_short", f"{hit_rate_short:.4f}"])
                w.writerow(["final_hit_rate_long", f"{hit_rate_long:.4f}"])
                
                # ========== 峰值记录 ==========
                w.writerow(["peak_margin", f"{self._src_peak_margin:.4f}"])
                w.writerow(["peak_source_conf", f"{self._peak_source_conf:.4f}"])
                w.writerow(["peak_source_conf_time", f"{self._peak_source_conf_time:.2f}" if self._peak_source_conf_time else "nan"])
                if self._src_peak_margin_pos is not None:
                    w.writerow(["peak_margin_pos_x", f"{self._src_peak_margin_pos[0]:.4f}"])
                    w.writerow(["peak_margin_pos_y", f"{self._src_peak_margin_pos[1]:.4f}"])
                    w.writerow(["peak_margin_pos_z", f"{self._src_peak_margin_pos[2]:.4f}"])
                else:
                    w.writerow(["peak_margin_pos_x", "nan"])
                    w.writerow(["peak_margin_pos_y", "nan"])
                    w.writerow(["peak_margin_pos_z", "nan"])
                
                # ========== SPIRAL 状态 ==========
                w.writerow(["spiral_substate", self._spiral_substate])
                w.writerow(["spiral_drift_scale", f"{self._spiral_drift_scale:.4f}"])
                w.writerow(["spiral_probe_scale", f"{self._spiral_probe_scale:.4f}"])
                w.writerow(["spiral_best_margin", f"{self._spiral_best_margin:.4f}"])
                w.writerow(["spiral_enter_count", self._spiral_enter_count])
                w.writerow(["spiral_exit_count", self._spiral_exit_count])
                w.writerow(["spiral_fail_count", self._spiral_fail_count])
                w.writerow(["spiral_no_improve_count", self._spiral_no_improve_count])
                
                # ========== 方向估计 ==========
                w.writerow(["dir_est_reach", f"{de.reach_dir:.4f}"])
                w.writerow(["dir_est_yaw", f"{de.yaw_dir:.4f}"])
                w.writerow(["dir_est_pitch", f"{de.pitch_dir:.4f}"])
                w.writerow(["dir_est_conf", f"{de.confidence:.4f}"])
                w.writerow(["dir_est_valid", de.valid])
                w.writerow(["dir_est_samples_lr", f"L{de.n_left}/R{de.n_right}"])
                w.writerow(["dir_est_samples_fb", f"F{de.n_forward}/B{de.n_backward}"])
                w.writerow(["dir_est_samples_ud", f"U{de.n_up}/D{de.n_down}"])
                
                # ========== 位置信息 ==========
                w.writerow(["final_sensor_x", f"{final_sensor[0]:.4f}"])
                w.writerow(["final_sensor_y", f"{final_sensor[1]:.4f}"])
                w.writerow(["final_sensor_z", f"{final_sensor[2]:.4f}"])
                w.writerow(["start_sensor_x", f"{start_sensor[0]:.4f}"])
                w.writerow(["start_sensor_y", f"{start_sensor[1]:.4f}"])
                w.writerow(["start_sensor_z", f"{start_sensor[2]:.4f}"])
                w.writerow(["final_ee_x", f"{final_ee[0]:.4f}"])
                w.writerow(["final_ee_y", f"{final_ee[1]:.4f}"])
                w.writerow(["final_ee_z", f"{final_ee[2]:.4f}"])
                w.writerow(["src_x", f"{self.src[0]:.4f}"])
                w.writerow(["src_y", f"{self.src[1]:.4f}"])
                w.writerow(["src_z", f"{self.src[2]:.4f}"])
                
                # ========== 位置偏差 ==========
                error_reach = final_sensor[0] - self.src[0] if not np.isnan(final_sensor[0]) else float('nan')
                error_yaw = final_sensor[1] - self.src[1] if not np.isnan(final_sensor[1]) else float('nan')
                error_pitch = final_sensor[2] - self.src[2] if not np.isnan(final_sensor[2]) else float('nan')
                w.writerow(["error_reach", f"{error_reach:.4f}"])
                w.writerow(["error_yaw", f"{error_yaw:.4f}"])
                w.writerow(["error_pitch", f"{error_pitch:.4f}"])
                
                # 峰值位置偏差
                if self._src_peak_margin_pos is not None:
                    peak_error_reach = self._src_peak_margin_pos[0] - self.src[0]
                    peak_error_yaw = self._src_peak_margin_pos[1] - self.src[1]
                    peak_error_pitch = self._src_peak_margin_pos[2] - self.src[2]
                    w.writerow(["peak_error_reach", f"{peak_error_reach:.4f}"])
                    w.writerow(["peak_error_yaw", f"{peak_error_yaw:.4f}"])
                    w.writerow(["peak_error_pitch", f"{peak_error_pitch:.4f}"])
                else:
                    w.writerow(["peak_error_reach", "nan"])
                    w.writerow(["peak_error_yaw", "nan"])
                    w.writerow(["peak_error_pitch", "nan"])
                
                # ========== 文件路径 ==========
                # ========== 其他关键指标 ==========
                w.writerow(["final_baseline", f"{self.baseline:.4f}"])
                w.writerow(["global_core_peak_margin", f"{self._global_core_peak_margin:.4f}"])
                if self._global_core_peak_pos is not None:
                    w.writerow(["global_core_peak_x", f"{self._global_core_peak_pos[0]:.4f}"])
                    w.writerow(["global_core_peak_y", f"{self._global_core_peak_pos[1]:.4f}"])
                    w.writerow(["global_core_peak_z", f"{self._global_core_peak_pos[2]:.4f}"])
                else:
                    w.writerow(["global_core_peak_x", "nan"])
                    w.writerow(["global_core_peak_y", "nan"])
                    w.writerow(["global_core_peak_z", "nan"])
                w.writerow(["startup_class", self._startup_class if self._startup_class else "unknown"])
                
                # ========== 文件路径 ==========
                w.writerow(["log_path", self.log_path])
                
        except Exception as e:
            self.get_logger().warn(f"summary write failed: {e}")


# ---------------------------------------------------------------------- #
# main
# ---------------------------------------------------------------------- #
def main(args=None):
    rclpy.init(args=args)
    node = GasSeekBio()
    try:
        rclpy.spin(node)
    except ExternalShutdownException:
        pass
    finally:
        try:
            if hasattr(node, "log_file"):
                node.log_file.flush()
                node.log_file.close()
        except Exception:
            pass
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()



