# Chapter 274: CNN LOB Prediction

## 1. Introduction — Convolutional Neural Networks for Limit Order Book Data

The Limit Order Book (LOB) is the fundamental data structure of modern electronic markets. At any given instant, the LOB records the outstanding buy (bid) and sell (ask) orders at discrete price levels. When we stack consecutive LOB snapshots along a time axis, the result is a two-dimensional matrix that bears a striking resemblance to a grayscale image: **rows correspond to price levels**, and **columns correspond to time steps**. This observation opens the door to applying Convolutional Neural Networks (CNNs) — the workhorse of computer vision — to financial microstructure prediction.

### Why CNNs for LOB?

Traditional feature engineering on LOB data manually constructs statistics such as bid–ask spread, order imbalance, or volume-weighted average price. CNNs replace this manual process with **learned spatial filters** that automatically extract informative patterns from the raw LOB representation:

- **Local patterns across price levels** — a 1D convolution sliding vertically across levels can detect supply–demand imbalances, walls of liquidity, or thin regions in the book.
- **Temporal patterns across snapshots** — a 1D convolution along the time axis can detect momentum, mean-reversion micro-signatures, or flickering quotes.
- **Joint spatio-temporal patterns** — a 2D convolution simultaneously captures how the shape of the book evolves over time, analogous to detecting edges and textures in images.

The seminal **DeepLOB** architecture (Zhang et al., 2019) demonstrated that a CNN–LSTM hybrid achieves state-of-the-art accuracy on mid-price movement prediction across multiple asset classes.

### LOB as an Image

Consider an LOB with `L` price levels on each side (bid and ask) and `T` consecutive snapshots. We construct a matrix of shape `(2L, T)` or, equivalently, a multi-channel tensor of shape `(C, L, T)` where channels encode price, volume, and derived features. This "LOB image" is the input to our CNN.

```
Time →  t₁   t₂   t₃  ...  tT
Level 1 (ask L) [ v  v  v  ...  v ]
Level 2         [ v  v  v  ...  v ]
  ...
Level L (ask 1) [ v  v  v  ...  v ]
Level L+1 (bid 1) [ v  v  v  ...  v ]
  ...
Level 2L (bid L) [ v  v  v  ...  v ]
```

Each cell contains the volume (or normalized log-volume) at that price level and time step.

---

## 2. Mathematical Foundations

### 2.1 One-Dimensional Convolution over Price Levels

Let **x** ∈ ℝ^(2L) be a single column of the LOB image (one snapshot). A 1D convolution with kernel **w** ∈ ℝ^k produces output:

$$
y_i = \sigma\!\left(\sum_{j=0}^{k-1} w_j \cdot x_{i+j} + b\right), \quad i = 0, 1, \ldots, 2L - k
$$

where σ is a nonlinear activation (e.g., ReLU), b is a bias term, and k is the kernel size. With **stride** s, we sample every s-th position:

$$
y_i = \sigma\!\left(\sum_{j=0}^{k-1} w_j \cdot x_{i \cdot s + j} + b\right)
$$

**Padding**: To preserve spatial dimensions, we pad the input with p zeros on each side, yielding output length ⌊(2L + 2p − k) / s⌋ + 1.

### 2.2 Two-Dimensional Convolution over LOB Snapshots

Given an LOB image **X** ∈ ℝ^(2L × T), a 2D kernel **W** ∈ ℝ^(kh × kw) produces:

$$
Y_{i,j} = \sigma\!\left(\sum_{m=0}^{k_h-1}\sum_{n=0}^{k_w-1} W_{m,n} \cdot X_{i+m,\, j+n} + b\right)
$$

This captures joint price-level and temporal patterns. In practice, we use multiple filters (F output channels), so the kernel tensor has shape (F, C_in, kh, kw) and the output has F channels.

### 2.3 Pooling Layers

**Max pooling** reduces spatial dimensions by taking the maximum over a local window:

$$
Y_{i,j} = \max_{0 \le m < p_h,\; 0 \le n < p_w} X_{i \cdot p_h + m,\; j \cdot p_w + n}
$$

This provides translational invariance and reduces computational cost. For LOB data, pooling along the time axis is more common than along price levels to preserve the ordered structure of the book.

### 2.4 Classification Head

After convolutional and pooling layers, the feature maps are flattened into a vector **z** ∈ ℝ^d and passed through fully connected layers:

$$
\hat{y} = \text{softmax}(W_2 \cdot \text{ReLU}(W_1 \cdot z + b_1) + b_2)
$$

producing a probability distribution over three classes: **Up**, **Down**, and **Stable** (for mid-price movement prediction).

The cross-entropy loss is:

$$
\mathcal{L} = -\sum_{c=1}^{3} y_c \log \hat{y}_c
$$

---

## 3. DeepLOB-Style Architecture

The DeepLOB architecture consists of three main blocks:

### Block 1: Convolutional Feature Extraction

1. **Conv2D** (32 filters, kernel 1×2, stride 1×2) — captures pairwise price–volume relationships
2. **Conv2D** (32 filters, kernel 4×1) — captures cross-level patterns
3. **Conv2D** (32 filters, kernel 4×1) — deeper cross-level patterns

Each convolution is followed by batch normalization and LeakyReLU activation.

### Block 2: Inception Module

An Inception-like module applies multiple kernel sizes in parallel (1×1, 3×1, 5×1) and concatenates the results, capturing multi-scale patterns across price levels simultaneously.

### Block 3: Temporal Aggregation

In the original DeepLOB, an LSTM processes the sequence of feature vectors. In our simplified CNN-only variant, we replace this with:
- **1D temporal convolutions** with dilated kernels
- **Global average pooling** along the time axis
- **Fully connected** classification head

### Architecture Summary

```
Input: (batch, 1, 2L, T)
  → Conv2D block (32 filters)
  → Conv2D block (32 filters)
  → Conv2D block (32 filters)
  → Inception module (multi-scale)
  → Temporal conv (64 filters)
  → Global average pool
  → FC(128) → ReLU → Dropout
  → FC(3) → Softmax
Output: P(up), P(down), P(stable)
```

---

## 4. Applications

### 4.1 Mid-Price Direction Prediction

The primary application is predicting the direction of mid-price movement over a short horizon (e.g., 10, 20, or 50 ticks ahead):

- **Label +1 (Up)**: mid-price increases by more than threshold α
- **Label −1 (Down)**: mid-price decreases by more than threshold α
- **Label 0 (Stable)**: mid-price stays within ±α

Typical prediction horizons range from milliseconds to seconds in high-frequency settings.

### 4.2 Spread Crossing Prediction

Predict whether the best bid will cross the current best ask (or vice versa), signaling imminent price movement.

### 4.3 Volume Imbalance Detection

CNNs can learn to detect transient volume imbalances across levels that precede large price moves, acting as an early warning system for institutional order flow.

### 4.4 Order Flow Toxicity

By analyzing the evolution of the LOB shape, CNNs can estimate the probability of informed trading activity (similar to VPIN but learned end-to-end).

---

## 5. Rust Implementation

Our Rust implementation provides a from-scratch CNN inference engine for LOB prediction. Key components:

### LOB Image Construction

The `LobImage` struct holds a 2D matrix of shape `(2 * levels, time_steps)` where each cell contains the normalized volume at that price level and time step. We fetch Bybit orderbook snapshots and stack them along the time axis.

### Convolutional Layers

We implement both 1D and 2D convolutional layers with configurable kernel size, stride, and padding. Weights are stored as `ndarray` arrays and the forward pass performs the convolution via nested loops (optimized for clarity, not HFT latency).

### Pooling and Classification

Max pooling reduces spatial dimensions. A fully connected head maps the flattened feature vector to three output classes via softmax.

### Bybit Integration

We use the Bybit v5 REST API to fetch live orderbook data for any trading pair, convert it to our LOB image format, and run inference.

See `rust/src/lib.rs` for the full implementation and `rust/examples/trading_example.rs` for a live trading demo.

---

## 6. Bybit Data Integration

### Orderbook Endpoint

```
GET https://api.bybit.com/v5/market/orderbook?category=spot&symbol=BTCUSDT&limit=50
```

Returns up to 200 levels of bid and ask data. We use the top `L` levels (e.g., 20) on each side.

### Data Normalization

Raw volumes are normalized per snapshot to remove absolute scale effects:

1. **Log transform**: v' = log(1 + v)
2. **Z-score normalization**: v'' = (v' − μ) / σ (computed over the current window)

### Snapshot Collection

To build an LOB image of width T, we collect T consecutive snapshots at a fixed polling interval (e.g., 100ms–1s). Each snapshot is a column in our image matrix.

### Real-Time Pipeline

```
Poll orderbook → Normalize → Append to ring buffer → Extract window → CNN inference → Signal
```

The ring buffer ensures constant-memory operation suitable for live trading.

---

## 7. Key Takeaways

1. **LOB as an image**: Treating the limit order book as a 2D matrix (levels × time) enables direct application of CNN architectures from computer vision.

2. **Spatial filters learn microstructure**: 1D convolutions across price levels automatically learn supply–demand imbalance features; 2D convolutions capture the joint evolution of the book over time.

3. **DeepLOB is the benchmark**: The DeepLOB architecture (CNN + LSTM) achieves state-of-the-art results on mid-price prediction. Our CNN-only variant trades some accuracy for simplicity and lower latency.

4. **Three-class classification is standard**: Predicting Up/Down/Stable with a softmax head and cross-entropy loss is the most common formulation for LOB prediction tasks.

5. **Normalization matters**: Log-transforming and z-scoring volumes is critical for stable training and generalization across different assets and market conditions.

6. **Pooling preserves order structure**: Max pooling along the time axis (but not across price levels) maintains the ordered nature of the book while reducing dimensionality.

7. **Latency vs. accuracy trade-off**: For live HFT, a pure CNN is preferred over CNN+LSTM due to lower inference latency. Our Rust implementation prioritizes clarity but can be optimized for production use.

8. **Data quality is paramount**: LOB data is noisy, with frequent cancellations and modifications. Robust preprocessing (handling missing levels, interpolation) is essential before feeding data to the CNN.

---

## References

- Zhang, Z., Zohren, S., & Roberts, S. (2019). *DeepLOB: Deep Convolutional Neural Networks for Limit Order Books*. IEEE Transactions on Signal Processing.
- Sirignano, J. (2019). *Deep Learning for Limit Order Books*. Quantitative Finance.
- Tsantekidis, A., et al. (2017). *Using Deep Learning to Detect Price Change Indications in Financial Markets*. European Signal Processing Conference.
- Passalis, N., et al. (2020). *Temporal Bag-of-Features Learning for Predicting Mid Price Movements Using High Frequency Limit Order Book Data*. IEEE Transactions on Emerging Topics in Computational Intelligence.
