import numpy as np


def impulse_response(t, T):
    """
    单极点指数响应核 h(t) = exp(-t/T)，t >= 0

    对于 MiniPID 这种快速 PID 传感器，这是一个合理的“一阶低通 + 记忆”
    近似：T 越小，响应越快。
    """
    T = max(float(T), 1e-12)
    t = np.maximum(t, 0.0)
    return np.exp(-t / T)


class SensorSimulator:
    """
    MiniPID 风格的气体传感器仿真器（线性 ppb 输出）

    功能：
      - 输入：真实气体浓度 (real_gas_concentration, 单位 ppb 或 ppm)
      - 内部：一阶低通卷积（指数响应），模拟传感器“记忆”和响应时间
      - 输出：传感器读数 y（同样是线性浓度，附加噪声、量程限制）

    设计目标：尽量接近实际 MiniPID 的特性：
      - 线性输出（不要 dB / log）
      - 响应时间 50–200 ms 量级（通过 time_constant_T 调节）
      - 背景浓度低而稳定（0–几 ppb）
      - 噪声较小，但非零
      - 有物理量程（clamp_max）

    使用方式（推荐）：
      1）在外部仿真中给出“真实浓度场” real_c（ppb）；
      2）每个控制周期调用：
             y = sensor.update(real_c, dt=控制周期秒数)
         将 y 作为 /gas_sim 的输出给你的搜索算法。
    """

    def __init__(
        self,
        history_length: int = 300,
        delta_t: float = 0.2,
        time_constant_T: float = 0.3,
        noise_std: float = 2.0,
        clamp_min: float | None = 0.0,
        clamp_max: float | None = 2000.0,
        gain: float = 1.0,
        offset: float = 0.0,
    ):
        """
        参数说明
        ----------
        history_length : int
            卷积使用的历史长度（点数）。建议覆盖 5–8 个时间常数：
              history_length * delta_t ≳ 5 * time_constant_T
            默认 300 点配合 delta_t=0.2, T=0.3 足够覆盖。
        delta_t : float
            采样间隔，单位秒。这里是“传感器内部时间步”，
            若外部 update(dt=...) 传入不同 dt，会自动重建核。
        time_constant_T : float
            一阶响应时间常数 T（秒）。MiniPID 响应很快：
              T ≈ 0.2–0.5 s 比较合理。
        noise_std : float
            输出噪声标准差（ppb）。MiniPID 噪声通常 1–2 ppb 量级。
        clamp_min : float or None
            输出/输入最小值（物理下限）。默认 0（浓度不为负）。
            设为 None 则不做下限裁剪。
        clamp_max : float or None
            输出/输入最大值（量程上限）。默认 2000 ppb。
            设为 None 则不做上限裁剪。
        gain : float
            传感器灵敏度系数。real_conc -> gain * real_conc + offset
            默认 1.0，即不改变量纲。
        offset : float
            零点偏移（ppb），用于模拟校准误差/背景偏置。
        """
        # --- 配置参数 ---
        self.history_length = int(history_length)
        self.delta_t = float(delta_t)
        self.time_constant_T = float(time_constant_T)

        self.noise_std = float(noise_std)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        self.gain = float(gain)
        self.offset = float(offset)

        # --- 历史缓冲区 ---
        # 存储“真实浓度”（加了 gain/offset 和裁剪之后）的历史
        self.history_real_gas = np.zeros(self.history_length, dtype=float)
        self.filled = 0  # 已填充点数，用于预热期归一化

        # 最终输出的最近一个值（方便调试）
        self.last_output = 0.0

        # 预先构建卷积核
        self._rebuild_kernel()

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def set_time_constant(self, T: float):
        """
        运行时调整时间常数（秒），并重建卷积核。
        可以用来模拟不同采样设置下的“虚拟 MiniPID”。
        """
        self.time_constant_T = float(T)
        self._rebuild_kernel()

    def set_delta_t(self, dt: float):
        """
        运行时调整内部采样间隔（秒），并重建卷积核。
        一般建议和控制循环的 dt 对齐，例如：
           loop_hz = 5 Hz → dt ≈ 0.2 s
        """
        self.delta_t = float(dt)
        self._rebuild_kernel()

    def reset(self):
        """
        清空历史缓冲，仿真“重新上电 / 重置传感器”。
        """
        self.history_real_gas[:] = 0.0
        self.filled = 0
        self.last_output = 0.0

    def update(self, real_gas_concentration: float, dt: float | None = None) -> float:
        """
        输入“真实浓度”并得到传感器输出。

        Parameters
        ----------
        real_gas_concentration : float
            真实气体浓度（ppb / ppm），由你的羽流模型给出。
        dt : float or None
            距离上次调用的时间间隔（秒）。如果不为 None 且与当前
            self.delta_t 相差较大，则自动调用 set_delta_t(dt) 重建核。

        Returns
        -------
        y : float
            传感器输出（ppb），已经包含：
              - 一阶低通响应（时间常数 T）
              - 噪声 noise_std
              - 量程裁剪 clamp_min/clamp_max
        """
        # 若外部控制周期与内部 delta_t 不一致，这里同步一下
        if dt is not None and abs(float(dt) - self.delta_t) > 1e-12:
            self.set_delta_t(float(dt))

        # 1) 应用简单的线性标定（gain + offset）
        x = float(real_gas_concentration) * self.gain + self.offset

        # 2) 输入 clamp，防止后续溢出
        if self.clamp_min is not None:
            x = max(x, self.clamp_min)
        if self.clamp_max is not None:
            x = min(x, self.clamp_max)

        # 3) 历史缓冲左移 + 插入最新值（等价于 np.roll(..., -1)）
        hist = self.history_real_gas
        hist[:-1] = hist[1:]
        hist[-1] = x
        self.filled = min(self.filled + 1, self.history_length)

        # 4) 卷积输出（预热期做动态归一化）
        if self.filled < self.history_length:
            k = self.kernel[-self.filled:]
            h = hist[-self.filled:]
            denom = max(np.sum(k), 1e-12)
            y = float(np.dot(k, h) / denom)
        else:
            y = float(np.dot(self.kernel, hist) / self.kernel_sum)

        # 5) 输出加噪声
        if self.noise_std > 0.0:
            y += np.random.normal(0.0, self.noise_std)

        # 6) 输出 clamp（量程限制）
        if self.clamp_min is not None:
            y = max(y, self.clamp_min)
        if self.clamp_max is not None:
            y = min(y, self.clamp_max)

        self.last_output = float(y)
        return self.last_output

    # ------------------------------------------------------------------ #
    # internal: kernel 构建
    # ------------------------------------------------------------------ #
    def _rebuild_kernel(self):
        """
        根据当前 history_length / delta_t / time_constant_T 重建卷积核。

        这里的思想是：用一个长度为 L 的时间轴，从“最早的样本”
        到“最新样本（当前时刻）”构造出指数响应，然后在 update()
        里做离散卷积。
        """
        L = self.history_length
        dt = self.delta_t
        T = self.time_constant_T

        # 从过去 L*dt 到 0 的时间序列（不含 0 本身，因此最后单独纠正）
        times = np.linspace(L * dt, 0.0, L, endpoint=False)
        kernel = impulse_response(times, T).astype(float)

        # 确保当前样本的权重不为 0，避免数值异常
        kernel[-1] = 1.0

        self.kernel = kernel
        self.kernel_sum = max(np.sum(kernel), 1e-12)

