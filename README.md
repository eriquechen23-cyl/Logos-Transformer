Logos-Transformer: Attention Is Love 🌌

A Linear Space-Time Miracle in a Subtraction-Free Universe.

Traditional Transformer architectures rely on the real number system (which includes negative numbers) and complex non-linear operations like Softmax. They treat "attention" as a competition for resources, leading to the infamous $O(N^2)$ quadratic complexity curse.

What if we completely abandon subtraction?

This project introduces the Logos-Transformer, a next-generation sequence modeling architecture built entirely within a purely positive universe (the Logos Universe). Here, there is no negation or cancellation—only the lossless superposition of energy, the filtering of will, and absolute resonance. By combining Parallel Prefix Sum, Chunking, and Local Learning, we achieve the physical limits of hardware optimization: easily processing sequences of 100,000+ tokens on a single consumer GPU without triggering Out of Memory (OOM) errors.

🌟 Core Philosophy & Mathematics

In the Logos Universe, all numbers $x$ strictly follow $x \in \mathbb{R}_{\ge 0} \cup \{\infty\}$.
The laws of mathematics are elevated into three sacred operators:

Pure Superposition ($+$): The lossless acceptance of energy. The historical state updates strictly through addition: $T_t = T_{t-1} + \tilde{X}_t$.

Hadamard Resonance ($\odot$): Element-wise dimensional alignment, replacing traditional matrix inner-product competition.

Logos Division ($\oslash$): The genesis axiom resolving nothingness and singularities:

$0 \oslash 0 = \mathbf{1}$ (Connection and foundation within the void)

$a \oslash 0 = \infty$ (The infinite outpouring of grace, bounded physically by max_grace)

Dual-Rail Features & Absolute Positive Time (LogosPE)

Dual-Rail Mapping: $f(x) = [\max(0, x), \max(0, -x)]$. Separates the "light" (positive) and "shadow" (negative) of the real number domain into independent positive channels.

LogosPE: Even dimensions follow $\sin^2(\theta)$, while odd dimensions follow $\cos^2(\theta)$. Time neither creates nor destroys energy; it only distributes it, ensuring absolute energy conservation: $\sin^2 + \cos^2 = 1$.

🚀 The Ultimate Optimization of Space-Time Complexity

The philosophical elegance of this architecture translates perfectly into revolutionary performance breakthroughs on hardware (CUDA):

Physical Characteristics

Traditional Transformer

Logos-Transformer (This Project)

Time (Training - Parallel)

$O(1)$

$O(\log N)$ (Uses Blelloch Scan to instantly settle history, destroying the RNN queue curse)

Time (Inference - Sequential)

$O(N \cdot d)$

$O(d)$ (Perfect linear inference speed)

Space (Training - VRAM)

$O(L \cdot N^2 + L \cdot N \cdot d)$

$O(C \cdot d)$ ★ (Ultimate compression via Chunking & Local Learning)

Space (Inference - KV Cache)

$O(N \cdot d)$

$O(d)$ ★ (Constant memory: History is losslessly compressed into the state space)

★ Freedom from Backpropagation (Learning Without Punishment):
This architecture completely discards loss.backward(). Truth is omnipresent. After calculating the alignment factor $\Delta = y \oslash \hat{y}$ at each layer, it immediately uses Multiplicative Weight Update (MWU) to grow or prune the "will" (weights). Once calculated, the computational graph is severed (del / .detach()), completely releasing the "karma" (memory) and locking GPU memory usage to a constant $O(C \cdot d)$.

💻 Quick Start

This project requires no complex dependencies. Built natively on PyTorch, it is highly suitable for running on Google Colab or Edge devices.

1. Installation & Execution

Assuming you have PyTorch installed, clone and run:

git clone [https://github.com/yourusername/Logos-Transformer.git](https://github.com/yourusername/Logos-Transformer.git)
cd Logos-Transformer
python logos_transformer_colab.py


2. Core Code Snippet

The following demonstrates how to train a massive sequence length of 100,000 with extremely low memory usage:

import torch
from logos_transformer_colab import dual_rail_map, logos_attention_chunking_train

# 1. Prepare an ultra-long sequence (e.g., N = 100,000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_real = torch.randn(2, 100000, 128, device=device)
y_real = torch.rand(2, 100000, 256, device=device) # Truth targets must be positive

# 2. Map to the Logos Dual-Rail Universe (Separating light and shadow)
X_logos = dual_rail_map(X_real)

# 3. Execute training (Chunking C=4096, completely avoiding OOM)
T_final, W_final = logos_attention_chunking_train(
    X_seq=X_logos, 
    y_seq=y_real, 
    chunk_size=4096, 
    lr=0.05
)


Example Output:

Begin Logos Training | Total Length N=100000 | Chunk Size C=4096 | Dimension d=256
------------------------------------------------------------
Chunk [000000 - 004096] settled | GPU memory severed and reset
Chunk [040960 - 045056] settled | GPU memory severed and reset
Chunk [081920 - 086016] settled | GPU memory severed and reset
------------------------------------------------------------
Training Complete! Total time: 0.21 seconds
Final Will Matrix W_raw_Q (first 5 dims): [1.3421, 0.9821, 1.0023, 1.1105, 0.8992]


📖 Documentation & Presentation

For detailed mathematical derivations, the logical closed-loop (Projection $\to$ Manifestation $\to$ Loss $\to$ Update), and the theological-physical metaphors, please refer to the included presentation slides:

attention_is_love.html (Open directly in your browser)

👨‍💻 Author

YE-LONG CHEN (陳燁龍)

AI Graduate Student | Student ID: P77141155

The inspiration for this architecture stems from the pursuit of mathematical beauty and the revelation of "Love and Grace" within Christian theology.

📜 License

This project is licensed under the MIT License - see the LICENSE file for details.