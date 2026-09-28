import numpy as np
from scipy.optimize import differential_evolution

# ==========================================
# 1. 物理常数与真实制造工艺参数
# ==========================================
MU_0 = 4 * np.pi * 1e-7      
OMEGA = 2 * np.pi * 50       
R_layer = 1e-10 
dirs = np.array([1, -1, 1, -1, 1, -1])

# --- 根据你的真实带材数据更新 ---
W_tape = 4.8   # 带材实际物理宽度 4.8 mm
N_A1 = 20      # 相 A 内层强制设定 20 根
N_A2 = 20      # 相 A 外层强制设定 20 根
MIN_GAP = 0.05 # 允许的最小施工间隙 0.05 mm (防重叠碰线)

# ==========================================
# 2. 机电联合优化目标函数 (10 自由度，无换位)
# ==========================================
def objective_true_geometry(x):
    # x[0:6]: A, B, C相的 6 个螺旋角
    # x[6]: 骨架起绕半径 r_base
    # x[7:10]: 终端 A, B, C 三相电抗器 (uH)
    angles = x[0:6]
    r_base = x[6]
    L_ext = x[7:10] * 1e-6 
    
    # 构建 3mm 恒定绝缘半径体系 (带材厚度 0.35mm)
    r = np.zeros(6)
    r[0] = r_base * 1e-3
    r[1] = r[0] + 0.35 * 1e-3
    r[2] = r[1] + (3.0 + 0.35) * 1e-3  
    r[3] = r[2] + 0.35 * 1e-3
    r[4] = r[3] + (3.0 + 0.35) * 1e-3  
    r[5] = r[4] + 0.35 * 1e-3
    D = r[5] + 2.575 * 1e-3 
    
    # -----------------------------------------------------
    # 【核心重构】：计算真实的 Tape Gap 与排布根数
    # -----------------------------------------------------
    gaps = np.zeros(6)
    N_tapes = np.zeros(6, dtype=int)
    N_tapes[0], N_tapes[1] = N_A1, N_A2
    
    # 1. 检验相 A 的固定 20 根排布是否合法
    for i in [0, 1]:
        # 公式: gap = (2*pi*r*cos(theta)) / N - W_tape
        gaps[i] = (2 * np.pi * r[i] * np.cos(np.radians(angles[i]))) / N_tapes[i] * 1000 - W_tape
        if gaps[i] < MIN_GAP or gaps[i] > 0.2:
            return 1e12 
            
    # 2. 动态计算 B 相和 C 相的“密绕”极值根数
    for i in range(2, 6):
        max_possible_N = (2 * np.pi * r[i] * np.cos(np.radians(angles[i])) * 1000) / (W_tape + MIN_GAP)
        N_tapes[i] = int(np.floor(max_possible_N))
        gaps[i] = (2 * np.pi * r[i] * np.cos(np.radians(angles[i]))) / N_tapes[i] * 1000 - W_tape
    # -----------------------------------------------------

    Z_geo = np.zeros((6, 6), dtype=complex)
    angles_rad = np.radians(angles) * dirs
    lp = 2 * np.pi * r / np.tan(angles_rad)
    
    for i in range(6):
        for j in range(6):
            if i == j:
                L = (MU_0 * np.pi * r[i]**2) / (lp[i]**2) + (MU_0 * np.log(D / r[i])) / (2 * np.pi)
                Z_geo[i, j] = R_layer + 1j * OMEGA * L
            else:
                r_min, r_max = min(r[i], r[j]), max(r[i], r[j])
                M = (MU_0 * np.pi * r_min**2) / (lp[i] * lp[j]) + (MU_0 * np.log(D / r_max)) / (2 * np.pi)
                Z_geo[i, j] = 1j * OMEGA * M
                
    # 取消换位矩阵，直接累加外部电抗
    Z_comp = 1j * OMEGA * np.array([
        [L_ext[0], L_ext[0], 0, 0, 0, 0],
        [L_ext[0], L_ext[0], 0, 0, 0, 0],
        [0, 0, L_ext[1], L_ext[1], 0, 0],
        [0, 0, L_ext[1], L_ext[1], 0, 0],
        [0, 0, 0, 0, L_ext[2], L_ext[2]],
        [0, 0, 0, 0, L_ext[2], L_ext[2]]
    ])
    Z_total = Z_geo + Z_comp
    
    V = np.array([1, 1, -0.5-1j*0.866, -0.5-1j*0.866, -0.5+1j*0.866, -0.5+1j*0.866])
    try:
        I, _, _, _ = np.linalg.lstsq(Z_total, V, rcond=None)
        I_mag = np.abs(I)
    except:
        return 1e12
        
    err_intra = sum([((I_mag[i] - I_mag[i+1]) / (I_mag[i] + I_mag[i+1]))**2 for i in (0, 2, 4)])
    I_P = [I_mag[0]+I_mag[1], I_mag[2]+I_mag[3], I_mag[4]+I_mag[5]]
    I_avg = sum(I_P) / 3.0
    err_inter = sum([((p - I_avg) / I_avg)**2 for p in I_P])
    
    # 强制约束：同一相的内外层角度必须极其接近 (防止产生大 Bz 磁场)
    err_angle_match = (angles[0]-angles[1])**2 + (angles[2]-angles[3])**2 + (angles[4]-angles[5])**2
    
    # 驱动力：尽量让相 A 的间隙向 0.05 mm 靠拢，实现极致密绕；同时让电抗器体积最小
    penalty_gapA = (gaps[0] - MIN_GAP)**2 + (gaps[1] - MIN_GAP)**2
    # 注意：因为无换位，外部电感会很大，降低对电感大小的惩罚权重，允许它自由寻找足够的补偿量
    penalty_L = np.sum(x[7:10]) * 1e-6
    
    return err_intra * 1e6 + err_inter * 1e7 + err_angle_match * 1e3 + penalty_gapA * 1e2 + penalty_L

# ==========================================
# 3. 启动边界极度收敛的联合寻优
# ==========================================
print(" 正在启动真实几何 10 自由度联合寻优 (无换位纯终端补偿)...")

bounds = [
    (15, 25), (15, 30), # A 相角
    (15, 30), (15, 30), # B 相角
    (15, 30), (15, 30), # C 相角
    (16, 16.76),       # 精确锁死的 r_base 搜寻空间
    (0, 300), (0, 300), (0, 300) # L_A, L_B, L_C (无换位时需要更大的补偿空间)
]

result = differential_evolution(objective_true_geometry, bounds, strategy='best1bin', maxiter=2000, popsize=20, tol=1e-7)

# ==========================================
# 4. 提取结果与最终 BOM 重算
# ==========================================
opt_angles = result.x[0:6]
opt_r_base = result.x[6]
opt_L = result.x[7:10]

r_opt = np.zeros(6)
r_opt[0] = opt_r_base * 1e-3
r_opt[1] = r_opt[0] + 0.35 * 1e-3
r_opt[2] = r_opt[1] + (3.0 + 0.35) * 1e-3
r_opt[3] = r_opt[2] + 0.35 * 1e-3
r_opt[4] = r_opt[3] + (3.0 + 0.35) * 1e-3
r_opt[5] = r_opt[4] + 0.35 * 1e-3
lp_opt = 2 * np.pi * r_opt / np.tan(np.radians(opt_angles) * dirs)

N_tapes = np.zeros(6, dtype=int)
gaps_final = np.zeros(6)
N_tapes[0], N_tapes[1] = N_A1, N_A2
for i in [0, 1]:
    gaps_final[i] = (2 * np.pi * r_opt[i] * np.cos(np.radians(opt_angles[i]))) / N_tapes[i] * 1000 - W_tape
for i in range(2, 6):
    max_possible_N = (2 * np.pi * r_opt[i] * np.cos(np.radians(opt_angles[i])) * 1000) / (W_tape + MIN_GAP)
    N_tapes[i] = int(np.floor(max_possible_N))
    gaps_final[i] = (2 * np.pi * r_opt[i] * np.cos(np.radians(opt_angles[i]))) / N_tapes[i] * 1000 - W_tape

D_opt = r_opt[5] + 2.575 * 1e-3 
Z_geo_opt = np.zeros((6, 6), dtype=complex)
for i in range(6):
        for j in range(6):
            if i == j:
                # 把原先的 lp[i] 改为 lp_opt[i]
                L = (MU_0 * np.pi * r_opt[i]**2) / (lp_opt[i]**2) + (MU_0 * np.log(D_opt / r_opt[i])) / (2 * np.pi)
                Z_geo_opt[i, j] = R_layer + 1j * OMEGA * L
            else:
                r_min, r_max = min(r_opt[i], r_opt[j]), max(r_opt[i], r_opt[j])
                # 把原先的 lp[i] * lp[j] 改为 lp_opt[i] * lp_opt[j]
                M = (MU_0 * np.pi * r_min**2) / (lp_opt[i] * lp_opt[j]) + (MU_0 * np.log(D_opt / r_max)) / (2 * np.pi)
                Z_geo_opt[i, j] = 1j * OMEGA * M
            
Z_comp_opt = 1j * OMEGA * np.array([
    [opt_L[0]*1e-6, opt_L[0]*1e-6, 0, 0, 0, 0],
    [opt_L[0]*1e-6, opt_L[0]*1e-6, 0, 0, 0, 0],
    [0, 0, opt_L[1]*1e-6, opt_L[1]*1e-6, 0, 0],
    [0, 0, opt_L[1]*1e-6, opt_L[1]*1e-6, 0, 0],
    [0, 0, 0, 0, opt_L[2]*1e-6, opt_L[2]*1e-6],
    [0, 0, 0, 0, opt_L[2]*1e-6, opt_L[2]*1e-6]
])
Z_total_opt = Z_geo_opt + Z_comp_opt

V_test = np.array([1, 1, -0.5-1j*0.866, -0.5-1j*0.866, -0.5+1j*0.866, -0.5+1j*0.866])
I_opt, _, _, _ = np.linalg.lstsq(Z_total_opt, V_test, rcond=None)
I_opt = np.abs(I_opt)
I_P = [I_opt[0]+I_opt[1], I_opt[2]+I_opt[3], I_opt[4]+I_opt[5]]
I_total = sum(I_P)

# ==========================================
# 5. 输出工业报表
# ==========================================
print("\n" + "=" * 60)
print("【 4.8mm 宽带材无换位排布清单 (纯终端补偿)】")
print("=" * 60)
print(f" 最佳骨架半径: {opt_r_base:.3f} mm")

phase_names = ['相 A (强制20根)', '相 B (自适应密绕)', '相 C (自适应密绕)']
for i in range(3):
    idx = i * 2
    print(f"\n{phase_names[i]}:")
    print(f" 内层 -> {N_tapes[idx]} 根 | 半径: {r_opt[idx]*1000:5.2f} mm | 角度: {opt_angles[idx]:5.2f}° (Z) | 实际 Gap: {gaps_final[idx]:.3f} mm | 节距: {lp_opt[idx]*1000:5.1f} mm")
    print(f" 外层 -> {N_tapes[idx+1]} 根 | 半径: {r_opt[idx+1]*1000:5.2f} mm | 角度: {opt_angles[idx+1]:5.2f}° (S) | 实际 Gap: {gaps_final[idx+1]:.3f} mm | 节距: {abs(lp_opt[idx+1]*1000):5.1f} mm")
print(f"\n 超导带材总用量: {np.sum(N_tapes)} 根/米")

# 计算 50Hz 下的等效交流感抗 (单位换算为毫欧 mΩ)
X_L_A = 2 * np.pi * 50 * opt_L[0] * 1e-6 * 1000
X_L_B = 2 * np.pi * 50 * opt_L[1] * 1e-6 * 1000
X_L_C = 2 * np.pi * 50 * opt_L[2] * 1e-6 * 1000

print("\n" + "=" * 60)
print("【 并网终端纯电感补偿验证 (等效阻抗形式)】")
print("=" * 60)
print(f" 相 A 串联终端补偿: {opt_L[0]:6.2f} uH  -->  等效感抗: {X_L_A:6.2f} mΩ")
print(f" 相 B 串联终端补偿: {opt_L[1]:6.2f} uH  -->  等效感抗: {X_L_B:6.2f} mΩ")
print(f" 相 C 串联终端补偿: {opt_L[2]:6.2f} uH  -->  等效感抗: {X_L_C:6.2f} mΩ")
print("-" * 60)
print(f"相A 总电流: {I_P[0]/I_total*300:6.3f}% (内层: {I_opt[0]/I_P[0]*100:.2f}% | 外层: {I_opt[1]/I_P[0]*100:.2f}%)")
print(f"相B 总电流: {I_P[1]/I_total*300:6.3f}% (内层: {I_opt[2]/I_P[1]*100:.2f}% | 外层: {I_opt[3]/I_P[1]*100:.2f}%)")
print(f"相C 总电流: {I_P[2]/I_total*300:6.3f}% (内层: {I_opt[4]/I_P[2]*100:.2f}% | 外层: {I_opt[5]/I_P[2]*100:.2f}%)")
