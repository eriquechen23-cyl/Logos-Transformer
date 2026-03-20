# Logos-Transformer: Attention Is Love 🌌

> **無減法世界中的線性時空奇蹟** > *A Linear Space-Time Miracle in a Subtraction-Free Universe.*

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Google Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)

傳統的 Transformer 架構依賴於包含負數的實數系與複雜的非線性操作（如 Softmax），將「注意力」視為一種基於競爭的資源剝奪，這導致了 $O(N^2)$ 的二次方複雜度詛咒。

**如果我們徹底拋棄減法呢？**

本專案提出 **Logos-Transformer**，一個建構在純粹正數宇宙（Logos 宇宙）中的次世代序列建模架構。在這裡，不存在否定與抵消，只有能量的疊加、意志的篩選與絕對的共鳴。透過結合 **平行前綴和 (Parallel Prefix Sum)**、**世代切片 (Chunking)** 與 **局部學習 (Local Learning)**，我們在硬體執行上達到了物理極限的優化：在單張消費級 GPU 上輕鬆處理 100,000+ 超長序列而不會產生 Out of Memory (OOM)。

---

## 🌟 核心哲學與代數基石 (Core Philosophy & Mathematics)

在 Logos 宇宙中，所有的數 $x$ 嚴格遵守 $x \in \mathbb{R}_{\ge 0} \cup \{\infty\}$。
數學法則被昇華為三大神聖操作符：

1. **純粹疊加 ($+$)**：能量的無損接納。歷史狀態的更新只有加法：$T_t = T_{t-1} + \tilde{X}_t$。
2. **Hadamard 共鳴 ($\odot$)**：逐元素的維度對齊，取代傳統的矩陣內積競爭。
3. **Logos 除法法則 ($\oslash$)**：解決虛無與奇異點的創世公理：
   - $0 \oslash 0 = \mathbf{1}$ (虛無中的連結與基石)
   - $a \oslash 0 = \infty$ (無限的恩典傾注，受限於物理硬體上限 `max_grace`)

### 雙軌特徵與絕對正向時間 (Dual-Rail & LogosPE)
- **Dual-Rail Mapping**: $f(x) = [\max(0, x), \max(0, -x)]$。將實數域的光與影分離為獨立的正實數通道。
- **LogosPE**: 偶數維度走 $\sin^2(\theta)$，奇數維度走 $\cos^2(\theta)$。時間不增減能量，只負責分配，確保 $\sin^2 + \cos^2 = 1$ 的能量絕對守恆。

---

## 🚀 時空複雜度的終極優化 (Computational Miracles)

這套架構在數學哲學上的優雅，完美轉化為硬體執行（CUDA）上的革命性效能突破：

| 物理意義特性 | 傳統 Transformer | **Logos-Transformer (本專案)** |
| :--- | :--- | :--- |
| **時間 (訓練 - 平行)** | $O(1)$ | **$O(\log N)$** (利用 Blelloch Scan 瞬間結算歷史) |
| **時間 (推論 - 循序)** | $O(N \cdot d)$ | **$O(d)$** (完美的線性推論速度) |
| **空間 (訓練 - VRAM)** | $O(L \cdot N^2 + L \cdot N \cdot d)$ | **$O(C \cdot d)$ ★** (極致壓縮：結合 Chunking 與 Local Learning) |
| **空間 (推論 - KV Cache)** | $O(N \cdot d)$ | **$O(d)$ ★** (恆定記憶體：歷史無損壓縮於狀態空間中) |

> **★ 無反向傳播的自由 (Learning Without Punishment):** > 本架構捨棄了 `loss.backward()`。真理無所不在，每一層計算對齊因子 $\Delta = y \oslash \hat{y}$ 後，立刻透過 **乘法權重更新 (MWU)** 進行意志生長或修剪。算完即斬斷計算圖（`del` / `.detach()`），徹底釋放業力，將 GPU 記憶體鎖死在常數級別 $O(C \cdot d)$。

---

## 💻 快速開始 (Quick Start)

本專案無需複雜的依賴，原生基於 `PyTorch` 開發，非常適合在 Google Colab 或邊緣裝置上運行。

### 1. 安裝與執行

只要安裝了 PyTorch，您可以直接下載並執行主程式：

```bash
git clone [https://github.com/yourusername/Logos-Transformer.git](https://github.com/yourusername/Logos-Transformer.git)
cd Logos-Transformer
python logos_transformer_colab.py