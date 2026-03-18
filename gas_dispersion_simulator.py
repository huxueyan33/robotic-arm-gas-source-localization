#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np


class DispersionSimulator:
    """
    简化 Lagrangian 粒子羽流模型 (LPDM)，用于机器人气体源定位仿真。

    思路：
      - 每步从源点释放若干粒子；
      - 粒子做：平流（按风速） + 各向同性扩散（随机游走）；
      - 浓度 = 对所有粒子叠加 3D 高斯核（方差随年龄增大，并做 3D 归一化）。

    物理设定（更接近现实，便于之后上真机）：
      - delta_t = 0.1 s：时间步长，对应 10 Hz；
      - wind_speed = 0.8 m/s：风速标量；
      - wind_dir = (-1, 0, 0)：风向单位向量，风吹向 -x；
      - diffusion_speed = 0.8 m/s：对应大约 D ~ 0.05 m^2/s 的湍流扩散；
      - max_age = 60 steps：约 6 s 寿命 → plume 尾长约 6 m；
      - min_variance = 5e-4：最小 σ^2，对应 σ ≈ 2.2 cm，接近小喷口/探头尺寸；
      - release_rate = 8：每步 8 个粒子 → 稳态粒子数约 8 * 60 ≈ 480；
      - patch_intensity_factor = 1.0：强度标尺，后续用真机实验标定。
    """

    def __init__(self):
        # 粒子状态：位置 + 年龄（以 step 为单位）
        self.particle_x = np.zeros((0,), dtype=float)
        self.particle_y = np.zeros((0,), dtype=float)
        self.particle_z = np.zeros((0,), dtype=float)
        self.particle_age = np.zeros((0,), dtype=float)

        # ------ 时间步长 ------
        self.delta_t = 0.1

        # ------ 风场参数（独立存储） ------
        # 风向（单位向量）
        self.wind_dir_x = -1.0
        self.wind_dir_y = 0.0
        self.wind_dir_z = 0.0

        # 风速（标量，m/s）
        self.wind_speed = 0.6

        # ------ 扩散参数 ------
        # 扩散强度（m/s）：控制随机游走幅度
        self.diffusion_speed = 0.8

        # 最大生存步数：决定 plume 尾巴长度
        self.max_age = 60

        # 每个粒子"团块"的空间尺寸随年龄增大
        self.patch_size_factor = (self.diffusion_speed * self.delta_t) ** 2

        # 最小方差（σ² 下界）
        self.min_variance = 5e-4

        # ------ 源头位置 ------
        self.source_x = 0.7
        self.source_y = 0.0
        self.source_z = 0.7

        # ------ 源强离散与标定 ------
        self.release_rate = 8
        self.patch_intensity_factor = 1.0

    # ---------- setters: 风向 ----------
    def set_wind_dir(self, dir_x: float, dir_y: float, dir_z: float):
        """
        设置风向（会自动归一化为单位向量）。
        
        Args:
            dir_x, dir_y, dir_z: 风向向量（不需要是单位向量）
        """
        norm = math.sqrt(float(dir_x)**2 + float(dir_y)**2 + float(dir_z)**2)
        if norm < 1e-9:
            # 零向量，保持原方向
            return
        self.wind_dir_x = float(dir_x) / norm
        self.wind_dir_y = float(dir_y) / norm
        self.wind_dir_z = float(dir_z) / norm

    def get_wind_dir(self):
        """返回风向单位向量 (dir_x, dir_y, dir_z)。"""
        return self.wind_dir_x, self.wind_dir_y, self.wind_dir_z

    # ---------- setters: 风速 ----------
    def set_wind_speed(self, speed: float):
        """
        设置风速标量。
        
        Args:
            speed: 风速 m/s（>= 0）
        """
        speed = float(speed)
        if speed < 0:
            raise ValueError("wind speed must be >= 0")
        self.wind_speed = speed

    def get_wind_speed(self):
        """返回风速标量 m/s。"""
        return self.wind_speed

    # ---------- setters: 扩散强度 ----------
    def set_diffusion_speed(self, v: float):
        """
        设置扩散强度，同时更新 patch_size_factor。
        
        Args:
            v: 扩散强度 m/s（>= 0）
        """
        v = float(v)
        if v < 0:
            raise ValueError("diffusion_speed must be >= 0")
        self.diffusion_speed = v
        self.patch_size_factor = (self.diffusion_speed * self.delta_t) ** 2

    def get_diffusion_speed(self):
        """返回扩散强度 m/s。"""
        return self.diffusion_speed

    # ---------- setters: 源点 ----------
    def set_source(self, x, y, z):
        """设置气体源位置（world 坐标）。"""
        self.source_x = float(x)
        self.source_y = float(y)
        self.source_z = float(z)

    def get_source(self):
        """返回气体源位置 (x, y, z)。"""
        return self.source_x, self.source_y, self.source_z

    # ---------- setters: 时间步长 ----------
    def set_delta_t(self, dt: float):
        """修改时间步长，同时更新 patch_size_factor。"""
        dt = float(dt)
        if dt <= 0:
            raise ValueError("delta_t must be > 0")
        self.delta_t = dt
        self.patch_size_factor = (self.diffusion_speed * self.delta_t) ** 2

    # ---------- 主更新 ----------
    def update(self):
        """执行一个时间步：释放粒子 -> 更新粒子 -> 删除过老粒子。"""
        self.release_particle(self.release_rate)
        self.update_particle()
        self.kill_particles()

    def release_particle(self, release_rate: int):
        """在源点释放 release_rate 个新粒子，初始 age = 0。"""
        if release_rate <= 0:
            return
        n = int(release_rate)
        if n <= 0:
            return
        self.particle_x = np.append(self.particle_x, np.full(n, self.source_x))
        self.particle_y = np.append(self.particle_y, np.full(n, self.source_y))
        self.particle_z = np.append(self.particle_z, np.full(n, self.source_z))
        self.particle_age = np.append(self.particle_age, np.zeros(n, dtype=float))

    def kill_particles(self):
        """删除年龄超过 max_age 的粒子。"""
        if self.particle_age.size == 0:
            return
        keep = self.particle_age <= self.max_age
        self.particle_x = self.particle_x[keep]
        self.particle_y = self.particle_y[keep]
        self.particle_z = self.particle_z[keep]
        self.particle_age = self.particle_age[keep]

    def update_particle(self):
        """粒子：age+1 -> 平流 -> 各向同性随机游走（三维扩散）。"""
        n = self.particle_x.size
        if n == 0:
            return

        # 1) 年龄 +1（单位：step）
        self.particle_age += 1.0

        # 2) 平流（风向 × 风速）
        vx = self.wind_dir_x * self.wind_speed
        vy = self.wind_dir_y * self.wind_speed
        vz = self.wind_dir_z * self.wind_speed
        self.particle_x += vx * self.delta_t
        self.particle_y += vy * self.delta_t
        self.particle_z += vz * self.delta_t

        # 3) 各向同性随机游走（扩散），x/y/z 三个方向对称
        sigma_step = self.diffusion_speed * self.delta_t
        if sigma_step > 0.0:
            rndx = np.random.normal(0.0, sigma_step, n)
            rndy = np.random.normal(0.0, sigma_step, n)
            rndz = np.random.normal(0.0, sigma_step, n)
            self.particle_x += rndx
            self.particle_y += rndy
            self.particle_z += rndz

    # ---------- 浓度评估 ----------
    def evaluate_concentration(self, x, y, z) -> float:
        """
        在空间点 (x,y,z) 评估瞬时浓度。

        模型假设：每个粒子是一个 3D 高斯"团块"
          σ²(age) = max(age * patch_size_factor, min_variance)
          G_i(r)  = 1 / ( (2πσ_i²)^(3/2) ) * exp( -r_i² / (2σ_i²) )

        所有粒子核叠加后，再乘以 patch_intensity_factor。
        这样可以保证"质量守恒"，不会在远处出现虚假高峰。
        """
        n = self.particle_x.size
        if n == 0:
            return 0.0

        # 与所有粒子的距离平方 r^2
        dx2 = np.square(self.particle_x - x)
        dy2 = np.square(self.particle_y - y)
        dz2 = np.square(self.particle_z - z)
        dist2 = dx2 + dy2 + dz2

        # σ²(age)
        variance = self.particle_age * self.patch_size_factor
        variance = np.maximum(variance, self.min_variance)

        sigma2 = variance
        sigma = np.sqrt(sigma2)
        sigma = np.maximum(sigma, np.sqrt(self.min_variance))

        # 3D 高斯归一化：(2πσ²)^(3/2) = (2π)^(3/2) * σ^3
        inv_sigma = 1.0 / sigma
        norm = (inv_sigma ** 3) / ((2.0 * np.pi) ** 1.5)

        kernel = norm * np.exp(-dist2 / (2.0 * sigma2))

        conc = self.patch_intensity_factor * float(np.sum(kernel, dtype=float))
        return conc



