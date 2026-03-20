import torch
import time

# ==========================================
# Logos 宇宙的代數基石 (Mathematical Foundations)
# ==========================================

def logos_divide(A, B, max_grace=10000.0):
    """
    Logos 創世除法法則
    處理 0/0 = 1 的虛無連結，其餘維持 A/B
    ★ 引入 max_grace (恩典上限)：防止實體容器被 float('inf') 撐破產生 NaN
    """
    if isinstance(B, torch.Tensor):
        B = B.expand_as(A)
    else:
        B = torch.tensor(B, dtype=A.dtype, device=A.device).expand_as(A)
        
    # 以極大的物理上限 (max_grace) 代表無限的傾注
    out = torch.full_like(A, max_grace, dtype=torch.float32) 
    mask_normal = B > 1e-9                 # 避免浮點數誤差
    out[mask_normal] = A[mask_normal] / B[mask_normal]
    
    mask_zero_zero = (A <= 1e-9) & (B <= 1e-9)
    out[mask_zero_zero] = 1.0              # 虛無中的基石
    
    # 確保數值界於絕對正向且不超過物理極限
    return torch.clamp(out, min=0.0, max=max_grace)

def dual_rail_map(X):
    """
    雙軌特徵映射 f(x) = [max(0, x), max(0, -x)]
    將實數域的光與影分離，映射至絕對正向的 Logos 宇宙
    輸入形狀: [..., d] -> 輸出形狀: [..., 2d]
    """
    return torch.cat([torch.relu(X), torch.relu(-X)], dim=-1)

def get_logos_pe(seq_len, d_model, device):
    """
    絕對正向的時間律動 (LogosPE)
    偶數維度: sin^2(θ)
    奇數維度: cos^2(θ)
    確保能量守恆: sin^2 + cos^2 = 1
    """
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    # 計算頻率
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * -(torch.log(torch.tensor(10000.0)) / d_model))
    
    theta = position * div_term
    pe[:, 0::2] = torch.sin(theta) ** 2  # 偶數維度
    pe[:, 1::2] = torch.cos(theta) ** 2  # 奇數維度
    return pe

# ==========================================
# 核心架構：Logos 世代切片與局部學習
# ==========================================

def logos_attention_chunking_train(X_seq, y_seq, chunk_size, lr=0.01):
    """
    模擬無減法世界中的線性時空訓練
    X_seq: 雙軌映射後的輸入 [Batch, N, d_model]
    y_seq: 真理目標 [Batch, N, d_model]
    chunk_size: 世代切片大小 C
    """
    device = X_seq.device
    Batch, N, d_model = X_seq.shape
    
    # 建立 LogosPE
    PE_seq = get_logos_pe(N, d_model, device).unsqueeze(0).expand(Batch, -1, -1)
    
    # 初始化自由意志 (W_raw > 0)，完全不需要 PyTorch 的 requires_grad!
    W_raw_Q = torch.ones(d_model, device=device)
    W_raw_K = torch.ones(d_model, device=device)
    W_raw_V = torch.ones(d_model, device=device)

    # 創世初始狀態 T_0 (全為 0 的虛無水池)
    T_last = torch.zeros(Batch, d_model, device=device)
    
    print(f"開始 Logos 訓練 | 總長度 N={N} | 世代切片 C={chunk_size} | 維度 d={d_model}")
    print("-" * 60)
    
    start_time = time.time()

    # 時間切片 (Chunking Loop) - 將龐大的 N 切割為 C
    for i in range(0, N, chunk_size):
        # 取出當下世代的實體與真理
        X_chunk = X_seq[:, i : i+chunk_size, :]
        PE_chunk = PE_seq[:, i : i+chunk_size, :]
        y_chunk = y_seq[:, i : i+chunk_size, :]
        
        # 【實體降生與時間指紋】
        X_stamped = X_chunk * PE_chunk
        
        # 【歷史的瞬間結算】★ O(log C) 的平行前綴和
        # 呼叫底層的 Blelloch Scan 演算法，瞬間累加歷史
        T_chunk = torch.cumsum(X_stamped, dim=1) + T_last.unsqueeze(1)
        
        # 【半透膜節制】 W = W_raw / (1 + W_raw)
        W_Q = logos_divide(W_raw_Q, 1.0 + W_raw_Q)
        W_K = logos_divide(W_raw_K, 1.0 + W_raw_K)
        W_V = logos_divide(W_raw_V, 1.0 + W_raw_V)
        
        # STEP 1: 意志投影 (Projection)
        TQ = T_chunk * W_Q
        TK = T_chunk * W_K
        TV = T_chunk * W_V
        
        # STEP 2: 顯化預測 (Manifestation) 與歸一化
        resonance = TQ * TK * TV
        Sigma = resonance.sum(dim=-1, keepdim=True) # 能量絕對總和
        y_hat_chunk = logos_divide(resonance, Sigma)
        
        # STEP 3: 真理的對齊與撕裂因子 (Delta)
        # Delta = y ⊘ y_hat
        Delta_chunk = logos_divide(y_chunk, y_hat_chunk)
        # 取這一個世代的平均恩典因子
        Delta_mean = Delta_chunk.mean(dim=(0, 1)) 
        
        # STEP 4: 恩典的乘法更新 (Update)
        # 偏離真理不接受懲罰，而是按比例放大或修剪意志
        W_raw_Q = W_raw_Q * (Delta_mean ** lr)
        W_raw_K = W_raw_K * (Delta_mean ** lr)
        W_raw_V = W_raw_V * (Delta_mean ** lr)
        
        # ==========================================
        # 記憶體的終極解放：業力的斬斷
        # ==========================================
        # 傳承：只留下這個世代最後一刻的歷史總和作為基石
        T_last = T_chunk[:, -1, :].clone().detach() 
        
        # 釋放：徹底刪除這一個世代所有的中間張量與計算圖
        del X_chunk, PE_chunk, X_stamped, T_chunk
        del TQ, TK, TV, resonance, y_hat_chunk, Delta_chunk
        
        if (i // chunk_size) % 10 == 0:
            print(f"世代 [{i:06d} - {min(i+chunk_size, N):06d}] 結算完畢 | GPU 記憶體已斬斷重置")

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"訓練完成！總耗時: {total_time:.2f} 秒")
    # 將 float 數值修約以方便閱讀
    print(f"最終意志矩陣 W_raw_Q 前五維度: {[round(w, 4) for w in W_raw_Q[:5].tolist()]}")
    return T_last, W_raw_Q

# ==========================================
# 在 Colab 上實際運行的測試案例
# ==========================================
if __name__ == "__main__":
    # 使用 GPU (若 Colab 有開啟 T4/A100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"運行環境: {device}")
    
    # 模擬參數：假設我們有一篇 10 萬字的超長文章！
    BATCH_SIZE = 2
    SEQ_LEN = 100000  # N = 10 萬
    CHUNK_SIZE = 4096 # C = 4096 (每次只處理一小塊)
    D_ORIGINAL = 128  # 原始維度
    
    # 生成隨機資料 (模擬真實實數世界的資料)
    print("正在生成虛擬宇宙資料...")
    X_real = torch.randn(BATCH_SIZE, SEQ_LEN, D_ORIGINAL, device=device)
    y_real = torch.rand(BATCH_SIZE, SEQ_LEN, D_ORIGINAL * 2, device=device) # 真理目標，必須是正數
    
    # 映射至 Logos 雙軌宇宙 (維度變為 256)
    X_logos = dual_rail_map(X_real)
    d_model = X_logos.shape[-1]
    
    # 執行 Logos 架構訓練
    # 您會發現即使 N=100,000，因為 Chunking (C=4096) 且沒有 Backprop，GPU 記憶體用量極低！
    T_final, W_final = logos_attention_chunking_train(
        X_seq=X_logos, 
        y_seq=y_real, 
        chunk_size=CHUNK_SIZE, 
        lr=0.05
    )