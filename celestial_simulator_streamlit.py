import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64
from PIL import Image
import time

# タイトルとイントロダクション
st.title('天体シミュレーター')
st.markdown("""
このアプリケーションは、ニュートンの運動法則と万有引力の法則に基づいて天体の動きや位置関係をシミュレーションするためのツールです。
太陽系の惑星運動から連星系、三体問題まで、様々な天体現象を視覚的に体験できます。
""")

# サイドバーの作成
st.sidebar.header('シミュレーション設定')

# 色名をHEX形式に変換する辞書
COLOR_MAP = {
    'red': '#FF0000',
    'green': '#00FF00',
    'blue': '#0000FF',
    'yellow': '#FFFF00',
    'orange': '#FFA500',
    'purple': '#800080',
    'brown': '#A52A2A',
    'gray': '#808080',
    'black': '#000000'
}

# 天体クラスの定義
class CelestialBody:
    """天体オブジェクトを表すクラス"""
    
    def __init__(self, name, mass, radius, position, velocity, color):
        """
        パラメータ:
        name (str): 天体の名前
        mass (float): 天体の質量 (kg)
        radius (float): 天体の半径 (m)
        position (array): 初期位置 [x, y, z] (m)
        velocity (array): 初期速度 [vx, vy, vz] (m/s)
        color (str): 天体の色
        """
        self.name = name
        self.mass = mass
        self.radius = radius
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.zeros(3)
        # 色名をHEX形式に変換
        self.color = COLOR_MAP.get(color, color) if isinstance(color, str) else color
        self.trajectory = [np.copy(self.position)]
        self.initial_position = np.copy(position)
        self.initial_velocity = np.copy(velocity)
        
    def update_trajectory(self):
        """現在位置を軌道履歴に追加"""
        self.trajectory.append(np.copy(self.position))
        
    def reset(self):
        """天体の位置と速度を初期状態に戻す"""
        self.position = np.copy(self.initial_position)
        self.velocity = np.copy(self.initial_velocity)
        self.acceleration = np.zeros(3)
        self.trajectory = [np.copy(self.position)]
        
    def __str__(self):
        return f"{self.name}: mass={self.mass}kg, position={self.position}m, velocity={self.velocity}m/s"

# 物理エンジンの定義
class PhysicsEngine:
    """天体の物理計算を行うエンジン"""
    
    def __init__(self, dt=86400, G=6.67430e-11):
        """
        パラメータ:
        dt (float): 時間ステップ (秒)
        G (float): 万有引力定数 (m^3 kg^-1 s^-2)
        """
        self.dt = dt
        self.G = G
        self.initial_dt = dt
        self.initial_G = G
        
    def calculate_acceleration(self, bodies):
        """
        全天体間の重力による加速度を計算
        
        パラメータ:
        bodies (list): CelestialBodyオブジェクトのリスト
        """
        # 全天体の加速度をゼロにリセット
        for body in bodies:
            body.acceleration = np.zeros(3)
        
        # 全天体ペアの重力相互作用を計算
        for i, body1 in enumerate(bodies):
            for body2 in bodies[i+1:]:
                # 2天体間のベクトルと距離を計算
                r_vec = body2.position - body1.position
                r = np.linalg.norm(r_vec)
                
                # 距離がゼロの場合（衝突など）はスキップ
                if r == 0:
                    continue
                
                # 万有引力の法則に基づく加速度を計算
                # F = G * m1 * m2 / r^2
                # a1 = F / m1 = G * m2 / r^2
                # a2 = F / m2 = G * m1 / r^2
                force_mag = self.G * body1.mass * body2.mass / (r * r)
                
                # 単位ベクトルを計算
                r_hat = r_vec / r
                
                # 各天体の加速度を更新（作用・反作用の法則）
                body1.acceleration += force_mag / body1.mass * r_hat
                body2.acceleration -= force_mag / body2.mass * r_hat
    
    def update_leapfrog(self, bodies):
        """
        リープフロッグ法による位置と速度の更新
        
        パラメータ:
        bodies (list): CelestialBodyオブジェクトのリスト
        """
        # 初期加速度を計算
        self.calculate_acceleration(bodies)
        
        # 全天体の速度を半ステップ更新
        for body in bodies:
            body.velocity += 0.5 * body.acceleration * self.dt
        
        # 全天体の位置を1ステップ更新
        for body in bodies:
            body.position += body.velocity * self.dt
            body.update_trajectory()
        
        # 新しい加速度を計算
        self.calculate_acceleration(bodies)
        
        # 全天体の速度を残りの半ステップ更新
        for body in bodies:
            body.velocity += 0.5 * body.acceleration * self.dt
            
    def reset(self):
        """物理エンジンのパラメータを初期状態に戻す"""
        self.dt = self.initial_dt
        self.G = self.initial_G

# 天体系クラスの定義
class CelestialSystem:
    """天体系を管理するクラス"""
    
    def __init__(self):
        """天体系の初期化"""
        self.bodies = []
        
    def add_body(self, body):
        """
        天体をシステムに追加
        
        パラメータ:
        body (CelestialBody): 追加する天体
        """
        self.bodies.append(body)
        
    def remove_body(self, body_name):
        """
        天体をシステムから削除
        
        パラメータ:
        body_name (str): 削除する天体の名前
        """
        self.bodies = [body for body in self.bodies if body.name != body_name]
        
    def get_body(self, body_name):
        """
        名前で天体を取得
        
        パラメータ:
        body_name (str): 取得する天体の名前
        
        戻り値:
        CelestialBody: 見つかった天体、見つからない場合はNone
        """
        for body in self.bodies:
            if body.name == body_name:
                return body
        return None
    
    def reset(self):
        """全天体を初期状態に戻す"""
        for body in self.bodies:
            body.reset()

# シミュレーションコントローラの定義
class SimulationController:
    """シミュレーションを制御するクラス"""
    
    def __init__(self, physics_engine, celestial_system):
        """
        パラメータ:
        physics_engine (PhysicsEngine): 物理計算エンジン
        celestial_system (CelestialSystem): 天体系
        """
        self.physics_engine = physics_engine
        self.celestial_system = celestial_system
        self.is_running = False
        self.time_scale = 1.0
        self.current_time = 0.0
        self.max_time = float('inf')
        
    def start(self):
        """シミュレーション開始"""
        self.is_running = True
        
    def pause(self):
        """シミュレーション一時停止"""
        self.is_running = False
        
    def reset(self):
        """シミュレーションリセット"""
        self.current_time = 0.0
        self.celestial_system.reset()
        self.physics_engine.reset()
        
    def set_time_scale(self, scale):
        """
        時間スケール調整
        
        パラメータ:
        scale (float): 新しい時間スケール
        """
        self.time_scale = scale
        
    def set_max_time(self, max_time):
        """
        最大シミュレーション時間を設定
        
        パラメータ:
        max_time (float): 最大シミュレーション時間 (秒)
        """
        self.max_time = max_time
        
    def step(self):
        """1ステップ進める"""
        if not self.is_running or self.current_time >= self.max_time:
            return False
        
        # 物理エンジンで天体の位置と速度を更新
        self.physics_engine.update_leapfrog(self.celestial_system.bodies)
        
        # 現在時間を更新
        self.current_time += self.physics_engine.dt * self.time_scale
        
        return True

# 可視化クラスの定義
class Visualizer:
    """シミュレーション結果を可視化するクラス"""
    
    def __init__(self, celestial_system, mode='2D'):
        """
        パラメータ:
        celestial_system (CelestialSystem): 天体系
        mode (str): 表示モード ('2D' または '3D')
        """
        self.celestial_system = celestial_system
        self.mode = mode
        self.fig = None
        self.ax = None
        self.show_trajectory = True
        self.trajectory_length = 100  # 表示する軌道の長さ
        self.show_labels = True
        self.size_scale = 1.0  # 天体サイズのスケール
        self.auto_scale = True  # 軸の自動スケーリング
        self.view_limits = None  # 表示範囲
        
    def initialize(self, figsize=(10, 8)):
        """
        描画環境の初期化
        
        パラメータ:
        figsize (tuple): 図のサイズ (幅, 高さ)
        """
        self.fig = plt.figure(figsize=figsize)
        
        if self.mode == '2D':
            self.ax = self.fig.add_subplot(111)
        else:  # '3D'
            self.ax = self.fig.add_subplot(111, projection='3d')
            
        # タイトルと軸ラベルを設定
        self.ax.set_title('天体シミュレーション')
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        if self.mode == '3D':
            self.ax.set_zlabel('Z [m]')
            
    def update(self):
        """フレーム更新"""
        self.ax.clear()
        
        # タイトルと軸ラベルを再設定
        self.ax.set_title('天体シミュレーション')
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        if self.mode == '3D':
            self.ax.set_zlabel('Z [m]')
        
        # 全天体を描画
        for body in self.celestial_system.bodies:
            # 天体の位置を描画
            size = max(20, body.radius/1e7) * self.size_scale
            if self.mode == '2D':
                self.ax.scatter(body.position[0], body.position[1], 
                               s=size, color=body.color, label=body.name if self.show_labels else None)
            else:  # '3D'
                self.ax.scatter(body.position[0], body.position[1], body.position[2], 
                               s=size, color=body.color, label=body.name if self.show_labels else None)
            
            # 軌道を描画
            if self.show_trajectory and len(body.trajectory) > 1:
                trajectory = np.array(body.trajectory[-self.trajectory_length:])
                if self.mode == '2D':
                    self.ax.plot(trajectory[:, 0], trajectory[:, 1], color=body.color, alpha=0.5)
                else:  # '3D'
                    self.ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                                color=body.color, alpha=0.5)
        
        # 凡例を表示（ラベル表示が有効な場合のみ）
        if self.show_labels:
            self.ax.legend()
        
        # 軸の範囲を設定
        if self.auto_scale:
            self.ax.autoscale(enable=True, axis='both', tight=True)
        elif self.view_limits is not None:
            self.ax.set_xlim(self.view_limits[0])
            self.ax.set_ylim(self.view_limits[1])
            if self.mode == '3D' and len(self.view_limits) > 2:
                self.ax.set_zlim(self.view_limits[2])
        
    def toggle_mode(self):
        """2D/3D表示切り替え"""
        if self.mode == '2D':
            self.mode = '3D'
        else:
            self.mode = '2D'
        
        # 図を再初期化
        plt.close(self.fig)
        self.initialize()
        
    def toggle_trajectory(self, show=None):
        """
        軌道表示切り替え
        
        パラメータ:
        show (bool): 軌道を表示するかどうか。Noneの場合は現在の状態を反転
        """
        if show is None:
            self.show_trajectory = not self.show_trajectory
        else:
            self.show_trajectory = show
            
    def toggle_labels(self, show=None):
        """
        ラベル表示切り替え
        
        パラメータ:
        show (bool): ラベルを表示するかどうか。Noneの場合は現在の状態を反転
        """
        if show is None:
            self.show_labels = not self.show_labels
        else:
            self.show_labels = show
            
    def set_trajectory_length(self, length):
        """
        表示する軌道の長さを設定
        
        パラメータ:
        length (int): 表示する軌道の長さ
        """
        self.trajectory_length = length
        
    def set_size_scale(self, scale):
        """
        天体サイズのスケールを設定
        
        パラメータ:
        scale (float): サイズスケール
        """
        self.size_scale = scale
        
    def set_auto_scale(self, auto_scale):
        """
        軸の自動スケーリングを設定
        
        パラメータ:
        auto_scale (bool): 自動スケーリングを有効にするかどうか
        """
        self.auto_scale = auto_scale
        
    def set_view_limits(self, xlim, ylim, zlim=None):
        """
        表示範囲を設定
        
        パラメータ:
        xlim (tuple): X軸の表示範囲 (min, max)
        ylim (tuple): Y軸の表示範囲 (min, max)
        zlim (tuple): Z軸の表示範囲 (min, max)（3Dモードのみ）
        """
        if zlim is not None:
            self.view_limits = (xlim, ylim, zlim)
        else:
            self.view_limits = (xlim, ylim)

# プリセットシナリオを作成する関数
def create_solar_system():
    """
    太陽系のシミュレーションを作成
    
    戻り値:
    CelestialSystem: 太陽系の天体系
    """
    system = CelestialSystem()
    
    # 太陽
    sun = CelestialBody(
        name="太陽",
        mass=1.989e30,  # kg
        radius=6.957e8,  # m
        position=[0, 0, 0],  # m
        velocity=[0, 0, 0],  # m/s
        color="#FFFF00"  # 黄色
    )
    system.add_body(sun)
    
    # 水星
    mercury = CelestialBody(
        name="水星",
        mass=3.301e23,  # kg
        radius=2.44e6,  # m
        position=[5.791e10, 0, 0],  # m
        velocity=[0, 4.74e4, 0],  # m/s
        color="#808080"  # グレー
    )
    system.add_body(mercury)
    
    # 金星
    venus = CelestialBody(
        name="金星",
        mass=4.867e24,  # kg
        radius=6.052e6,  # m
        position=[1.082e11, 0, 0],  # m
        velocity=[0, 3.5e4, 0],  # m/s
        color="#FFA500"  # オレンジ
    )
    system.add_body(venus)
    
    # 地球
    earth = CelestialBody(
        name="地球",
        mass=5.972e24,  # kg
        radius=6.371e6,  # m
        position=[1.496e11, 0, 0],  # m
        velocity=[0, 2.98e4, 0],  # m/s
        color="#0000FF"  # 青
    )
    system.add_body(earth)
    
    # 火星
    mars = CelestialBody(
        name="火星",
        mass=6.417e23,  # kg
        radius=3.39e6,  # m
        position=[2.279e11, 0, 0],  # m
        velocity=[0, 2.41e4, 0],  # m/s
        color="#FF0000"  # 赤
    )
    system.add_body(mars)
    
    # 木星
    jupiter = CelestialBody(
        name="木星",
        mass=1.898e27,  # kg
        radius=6.991e7,  # m
        position=[7.786e11, 0, 0],  # m
        velocity=[0, 1.31e4, 0],  # m/s
        color="#A52A2A"  # 茶色
    )
    system.add_body(jupiter)
    
    return system


def create_earth_moon_system():
    """
    地球-月系のシミュレーションを作成
    
    戻り値:
    CelestialSystem: 地球-月系の天体系
    """
    system = CelestialSystem()
    
    # 地球
    earth = CelestialBody(
        name="地球",
        mass=5.972e24,  # kg
        radius=6.371e6,  # m
        position=[0, 0, 0],  # m
        velocity=[0, 0, 0],  # m/s
        color="#0000FF"  # 青
    )
    system.add_body(earth)
    
    # 月
    moon = CelestialBody(
        name="月",
        mass=7.342e22,  # kg
        radius=1.737e6,  # m
        position=[3.844e8, 0, 0],  # m
        velocity=[0, 1.022e3, 0],  # m/s
        color="#808080"  # グレー
    )
    system.add_body(moon)
    
    return system


def create_binary_star_system():
    """
    連星系のシミュレーションを作成
    
    戻り値:
    CelestialSystem: 連星系の天体系
    """
    system = CelestialSystem()
    
    # 恒星1
    star1 = CelestialBody(
        name="恒星1",
        mass=1.5e30,  # kg
        radius=7e8,  # m
        position=[3e11, 0, 0],  # m
        velocity=[0, 2e4, 0],  # m/s
        color="#FFFF00"  # 黄色
    )
    system.add_body(star1)
    
    # 恒星2
    star2 = CelestialBody(
        name="恒星2",
        mass=1.0e30,  # kg
        radius=5e8,  # m
        position=[-3e11, 0, 0],  # m
        velocity=[0, -3e4, 0],  # m/s
        color="#FFA500"  # オレンジ
    )
    system.add_body(star2)
    
    return system


def create_three_body_system():
    """
    三体問題のシミュレーションを作成
    
    戻り値:
    CelestialSystem: 三体系の天体系
    """
    system = CelestialSystem()
    
    # 天体1
    body1 = CelestialBody(
        name="天体1",
        mass=1.0e30,  # kg
        radius=5e8,  # m
        position=[3e11, 0, 0],  # m
        velocity=[0, 2e4, 0],  # m/s
        color="#FF0000"  # 赤
    )
    system.add_body(body1)
    
    # 天体2
    body2 = CelestialBody(
        name="天体2",
        mass=1.0e30,  # kg
        radius=5e8,  # m
        position=[-3e11, 0, 0],  # m
        velocity=[0, -2e4, 0],  # m/s
        color="#0000FF"  # 青
    )
    system.add_body(body2)
    
    # 天体3
    body3 = CelestialBody(
        name="天体3",
        mass=5.0e29,  # kg
        radius=3e8,  # m
        position=[0, 4e11, 0],  # m
        velocity=[-1.5e4, 0, 0],  # m/s
        color="#00FF00"  # 緑
    )
    system.add_body(body3)
    
    return system

# アニメーションをGIFに変換する関数
def create_animation_gif(simulation_controller, visualizer, frames=100, interval=50):
    """
    アニメーションGIFを作成
    
    パラメータ:
    simulation_controller (SimulationController): シミュレーション制御オブジェクト
    visualizer (Visualizer): 可視化オブジェクト
    frames (int): アニメーションのフレーム数
    interval (int): フレーム間の時間 (ミリ秒)
    
    戻り値:
    bytes: GIFデータ
    """
    # シミュレーションを初期状態にリセット
    simulation_controller.reset()
    simulation_controller.start()
    
    # フレームを格納するリスト
    frame_images = []
    
    # 各フレームを生成
    for _ in range(frames):
        # シミュレーションを1ステップ進める
        simulation_controller.step()
        
        # 描画を更新
        visualizer.update()
        
        # 図をバイトストリームに変換
        buf = io.BytesIO()
        visualizer.fig.savefig(buf, format='png')
        buf.seek(0)
        
        # PILイメージに変換
        img = Image.open(buf)
        frame_images.append(img)
    
    # GIFを作成
    gif_buf = io.BytesIO()
    frame_images[0].save(
        gif_buf, 
        format='GIF', 
        save_all=True, 
        append_images=frame_images[1:], 
        duration=interval, 
        loop=0
    )
    gif_buf.seek(0)
    
    return gif_buf.getvalue()

# Streamlitアプリのメイン部分
def main():
    # セッション状態の初期化
    if 'celestial_system' not in st.session_state:
        st.session_state.celestial_system = create_solar_system()
    
    if 'physics_engine' not in st.session_state:
        st.session_state.physics_engine = PhysicsEngine(dt=86400)  # 1日のステップ
    
    if 'simulation_controller' not in st.session_state:
        st.session_state.simulation_controller = SimulationController(
            st.session_state.physics_engine, 
            st.session_state.celestial_system
        )
    
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = Visualizer(st.session_state.celestial_system, mode='2D')
        st.session_state.visualizer.initialize()
    
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    
    if 'current_scenario' not in st.session_state:
        st.session_state.current_scenario = 'solar_system'
    
    # シナリオ選択
    scenario = st.sidebar.radio(
        "シナリオ選択",
        ('太陽系', '地球-月系', '連星系', '三体問題', 'カスタム'),
        index=0 if st.session_state.current_scenario == 'solar_system' else
               1 if st.session_state.current_scenario == 'earth_moon' else
               2 if st.session_state.current_scenario == 'binary_star' else
               3 if st.session_state.current_scenario == 'three_body' else 4
    )
    
    # シナリオが変更された場合
    scenario_map = {
        '太陽系': 'solar_system',
        '地球-月系': 'earth_moon',
        '連星系': 'binary_star',
        '三体問題': 'three_body',
        'カスタム': 'custom'
    }
    
    if st.session_state.current_scenario != scenario_map[scenario]:
        if scenario == '太陽系':
            st.session_state.celestial_system = create_solar_system()
        elif scenario == '地球-月系':
            st.session_state.celestial_system = create_earth_moon_system()
        elif scenario == '連星系':
            st.session_state.celestial_system = create_binary_star_system()
        elif scenario == '三体問題':
            st.session_state.celestial_system = create_three_body_system()
        # カスタムの場合は何もしない
        
        # シミュレーションコントローラと可視化を更新
        st.session_state.simulation_controller = SimulationController(
            st.session_state.physics_engine, 
            st.session_state.celestial_system
        )
        st.session_state.visualizer.celestial_system = st.session_state.celestial_system
        
        # 現在のシナリオを更新
        st.session_state.current_scenario = scenario_map[scenario]
    
    # シミュレーション設定
    st.sidebar.subheader('シミュレーション設定')
    
    # 時間ステップ
    dt_days = st.sidebar.slider('時間ステップ (日)', 0.1, 10.0, 1.0)
    st.session_state.physics_engine.dt = dt_days * 86400  # 日から秒に変換
    
    # 時間スケール
    time_scale = st.sidebar.slider('時間スケール', 0.1, 10.0, 1.0)
    st.session_state.simulation_controller.set_time_scale(time_scale)
    
    # 表示設定
    st.sidebar.subheader('表示設定')
    
    # 2D/3D表示
    display_mode = st.sidebar.radio('表示モード', ('2D', '3D'))
    if st.session_state.visualizer.mode != display_mode:
        st.session_state.visualizer.mode = display_mode
        plt.close(st.session_state.visualizer.fig)
        st.session_state.visualizer.initialize()
    
    # 軌道表示
    show_trajectory = st.sidebar.checkbox('軌道を表示', value=True)
    st.session_state.visualizer.toggle_trajectory(show_trajectory)
    
    # ラベル表示
    show_labels = st.sidebar.checkbox('ラベルを表示', value=True)
    st.session_state.visualizer.toggle_labels(show_labels)
    
    # 軌道の長さ
    trajectory_length = st.sidebar.slider('軌道の長さ', 10, 1000, 100)
    st.session_state.visualizer.set_trajectory_length(trajectory_length)
    
    # 天体サイズ
    size_scale = st.sidebar.slider('天体サイズ', 0.1, 10.0, 1.0)
    st.session_state.visualizer.set_size_scale(size_scale)
    
    # シミュレーション制御ボタン
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button('開始'):
            st.session_state.is_running = True
            st.session_state.simulation_controller.start()
    
    with col2:
        if st.button('一時停止'):
            st.session_state.is_running = False
            st.session_state.simulation_controller.pause()
    
    with col3:
        if st.button('リセット'):
            st.session_state.is_running = False
            st.session_state.simulation_controller.reset()
    
    # 天体パラメータの表示と編集
    st.subheader('天体パラメータ')
    
    # 天体リストのタブ
    body_tabs = st.tabs([body.name for body in st.session_state.celestial_system.bodies] + ["新規追加"])
    
    # 既存の天体の編集
    for i, tab in enumerate(body_tabs[:-1]):  # 最後のタブ（新規追加）を除く
        with tab:
            body = st.session_state.celestial_system.bodies[i]
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_name = st.text_input('名前', value=body.name, key=f'name_{i}')
                new_mass = st.number_input('質量 (kg)', value=float(body.mass), format='%e', key=f'mass_{i}')
                new_radius = st.number_input('半径 (m)', value=float(body.radius), format='%e', key=f'radius_{i}')
                # 色選択をHEX形式に統一
                new_color = st.color_picker('色', value=body.color, key=f'color_{i}')
            
            with col2:
                st.write('位置 (m)')
                new_pos_x = st.number_input('X', value=float(body.position[0]), format='%e', key=f'pos_x_{i}')
                new_pos_y = st.number_input('Y', value=float(body.position[1]), format='%e', key=f'pos_y_{i}')
                new_pos_z = st.number_input('Z', value=float(body.position[2]), format='%e', key=f'pos_z_{i}')
                
                st.write('速度 (m/s)')
                new_vel_x = st.number_input('X', value=float(body.velocity[0]), format='%e', key=f'vel_x_{i}')
                new_vel_y = st.number_input('Y', value=float(body.velocity[1]), format='%e', key=f'vel_y_{i}')
                new_vel_z = st.number_input('Z', value=float(body.velocity[2]), format='%e', key=f'vel_z_{i}')
            
            # 更新ボタン
            if st.button('更新', key=f'update_{i}'):
                body.name = new_name
                body.mass = new_mass
                body.radius = new_radius
                body.color = new_color
                body.position = np.array([new_pos_x, new_pos_y, new_pos_z])
                body.velocity = np.array([new_vel_x, new_vel_y, new_vel_z])
                body.initial_position = np.copy(body.position)
                body.initial_velocity = np.copy(body.velocity)
                st.success(f'天体 "{new_name}" を更新しました。')
            
            # 削除ボタン
            if st.button('削除', key=f'delete_{i}'):
                st.session_state.celestial_system.remove_body(body.name)
                st.success(f'天体 "{body.name}" を削除しました。')
                st.experimental_rerun()
    
    # 新規天体の追加
    with body_tabs[-1]:
        col1, col2 = st.columns(2)
        
        with col1:
            new_name = st.text_input('名前', value='新しい天体', key='new_name')
            new_mass = st.number_input('質量 (kg)', value=1.0e24, format='%e', key='new_mass')
            new_radius = st.number_input('半径 (m)', value=6.0e6, format='%e', key='new_radius')
            # 色選択をHEX形式に統一
            new_color = st.color_picker('色', value='#00FF00', key='new_color')
        
        with col2:
            st.write('位置 (m)')
            new_pos_x = st.number_input('X', value=2.0e11, format='%e', key='new_pos_x')
            new_pos_y = st.number_input('Y', value=0.0, format='%e', key='new_pos_y')
            new_pos_z = st.number_input('Z', value=0.0, format='%e', key='new_pos_z')
            
            st.write('速度 (m/s)')
            new_vel_x = st.number_input('X', value=0.0, format='%e', key='new_vel_x')
            new_vel_y = st.number_input('Y', value=2.0e4, format='%e', key='new_vel_y')
            new_vel_z = st.number_input('Z', value=0.0, format='%e', key='new_vel_z')
        
        # 追加ボタン
        if st.button('追加', key='add_new_body'):
            # 同名の天体がないか確認
            if any(body.name == new_name for body in st.session_state.celestial_system.bodies):
                st.error(f'"{new_name}" という名前の天体は既に存在します。')
            else:
                new_body = CelestialBody(
                    name=new_name,
                    mass=new_mass,
                    radius=new_radius,
                    position=[new_pos_x, new_pos_y, new_pos_z],
                    velocity=[new_vel_x, new_vel_y, new_vel_z],
                    color=new_color
                )
                st.session_state.celestial_system.add_body(new_body)
                st.success(f'新しい天体 "{new_name}" を追加しました。')
                st.experimental_rerun()
    
    # シミュレーション表示
    st.subheader('シミュレーション')
    
    # プレースホルダーを作成
    simulation_placeholder = st.empty()
    
    # シミュレーションを実行
    if st.session_state.is_running:
        # 一定回数のステップを実行
        for _ in range(10):  # 10ステップごとに表示を更新
            st.session_state.simulation_controller.step()
    
    # 可視化を更新
    st.session_state.visualizer.update()
    
    # 図を表示
    simulation_placeholder.pyplot(st.session_state.visualizer.fig)
    
    # アニメーションGIFの生成ボタン
    st.subheader('アニメーションGIF生成')
    
    col1, col2 = st.columns(2)
    
    with col1:
        frames = st.number_input('フレーム数', min_value=10, max_value=500, value=100, key='gif_frames')
    
    with col2:
        interval = st.number_input('フレーム間隔 (ミリ秒)', min_value=10, max_value=500, value=50, key='gif_interval')
    
    if st.button('アニメーションGIFを生成', key='generate_gif'):
        with st.spinner('アニメーションを生成中...'):
            # 現在の状態を保存
            current_running = st.session_state.is_running
            
            # アニメーションを生成
            gif_data = create_animation_gif(
                st.session_state.simulation_controller,
                st.session_state.visualizer,
                frames=frames,
                interval=interval
            )
            
            # Base64エンコード
            b64 = base64.b64encode(gif_data).decode()
            
            # HTMLで表示
            st.markdown(
                f'<img src="data:image/gif;base64,{b64}" alt="animation">',
                unsafe_allow_html=True
            )
            
            # ダウンロードボタン
            st.download_button(
                label="GIFをダウンロード",
                data=gif_data,
                file_name="celestial_simulation.gif",
                mime="image/gif",
                key='download_gif'
            )
            
            # 元の状態に戻す
            if current_running:
                st.session_state.is_running = True
                st.session_state.simulation_controller.start()
            else:
                st.session_state.is_running = False
                st.session_state.simulation_controller.pause()
    
    # Streamlit Cloudへのデプロイ方法
    st.subheader('Streamlit Cloudへのデプロイ方法')
    
    st.markdown("""
    このアプリケーションをStreamlit Cloudにデプロイするには、以下の手順に従ってください：
    
    1. GitHubアカウントを作成し、新しいリポジトリを作成します。
    2. このコードを `app.py` として保存し、リポジトリにアップロードします。
    3. `requirements.txt` ファイルを作成し、以下の内容を記述します：
       ```
       streamlit>=1.22.0
       numpy>=1.22.0
       matplotlib>=3.5.0
       pillow>=9.0.0
       ```
    4. [Streamlit Cloud](https://streamlit.io/cloud) にアクセスし、GitHubアカウントでログインします。
    5. 「New app」をクリックし、作成したリポジトリを選択します。
    6. メインファイルとして `app.py` を指定し、「Deploy」をクリックします。
    
    数分後、アプリケーションがデプロイされ、公開URLが提供されます。Streamlit Cloudの無料枠では、このようなアプリケーションを公開することができます。
    """)
    
    # フッター
    st.markdown('---')
    st.markdown('© 2025 天体シミュレーター')

if __name__ == '__main__':
    main()
