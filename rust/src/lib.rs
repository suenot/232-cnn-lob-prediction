//! CNN LOB Prediction
//!
//! Convolutional Neural Network for Limit Order Book mid-price direction prediction.
//! Implements a DeepLOB-inspired architecture with 1D/2D convolutions, max pooling,
//! and a classification head for Up/Down/Stable prediction.

use ndarray::{Array1, Array2, Array4, s};
use rand::Rng;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// LOB Image representation
// ---------------------------------------------------------------------------

/// A Limit Order Book image: rows = 2*levels (ask levels + bid levels),
/// columns = time steps.  Each cell holds normalised log-volume.
#[derive(Debug, Clone)]
pub struct LobImage {
    /// Number of price levels on each side (bid/ask).
    pub levels: usize,
    /// Number of time steps (snapshots).
    pub time_steps: usize,
    /// Data matrix of shape (2*levels, time_steps).
    pub data: Array2<f64>,
}

impl LobImage {
    /// Create a new empty LOB image.
    pub fn new(levels: usize, time_steps: usize) -> Self {
        Self {
            levels,
            time_steps,
            data: Array2::zeros((2 * levels, time_steps)),
        }
    }

    /// Set a column (one snapshot) of the LOB image.
    /// `asks` and `bids` should each have length `self.levels`.
    pub fn set_snapshot(&mut self, t: usize, asks: &[f64], bids: &[f64]) {
        assert_eq!(asks.len(), self.levels);
        assert_eq!(bids.len(), self.levels);
        for i in 0..self.levels {
            self.data[[i, t]] = asks[i];
        }
        for i in 0..self.levels {
            self.data[[self.levels + i, t]] = bids[i];
        }
    }

    /// Normalise volumes: log(1+v) then z-score per snapshot.
    pub fn normalize(&mut self) {
        // Log transform
        self.data.mapv_inplace(|v| (1.0 + v.abs()).ln());

        // Z-score per column (snapshot)
        for t in 0..self.time_steps {
            let col = self.data.slice(s![.., t]).to_owned();
            let n = col.len() as f64;
            let mean = col.sum() / n;
            let var = col.mapv(|x| (x - mean).powi(2)).sum() / n;
            let std = var.sqrt().max(1e-8);
            for i in 0..2 * self.levels {
                self.data[[i, t]] = (self.data[[i, t]] - mean) / std;
            }
        }
    }

    /// Convert to a 4-D tensor (batch=1, channels=1, height=2*levels, width=time_steps)
    /// suitable for 2-D convolution.
    pub fn to_tensor(&self) -> Array4<f64> {
        let h = 2 * self.levels;
        let w = self.time_steps;
        let mut tensor = Array4::zeros((1, 1, h, w));
        for i in 0..h {
            for j in 0..w {
                tensor[[0, 0, i, j]] = self.data[[i, j]];
            }
        }
        tensor
    }
}

// ---------------------------------------------------------------------------
// 1-D Convolutional Layer (over a single axis)
// ---------------------------------------------------------------------------

/// 1-D convolutional layer operating on a 1-D signal.
#[derive(Debug, Clone)]
pub struct Conv1D {
    /// Kernel weights of shape (out_channels, in_channels, kernel_size).
    pub weights: Vec<Vec<Vec<f64>>>,
    /// Bias per output channel.
    pub biases: Vec<f64>,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub out_channels: usize,
    pub in_channels: usize,
}

impl Conv1D {
    /// Create a new Conv1D layer with random (Xavier) initialisation.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (in_channels * kernel_size) as f64).sqrt();
        let weights = (0..out_channels)
            .map(|_| {
                (0..in_channels)
                    .map(|_| {
                        (0..kernel_size)
                            .map(|_| rng.gen_range(-scale..scale))
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let biases = vec![0.0; out_channels];
        Self {
            weights,
            biases,
            kernel_size,
            stride,
            padding,
            out_channels,
            in_channels,
        }
    }

    /// Forward pass.  Input shape: (in_channels, length).
    /// Output shape: (out_channels, output_length).
    pub fn forward(&self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        assert_eq!(input.len(), self.in_channels);
        let input_len = input[0].len();
        let padded_len = input_len + 2 * self.padding;
        let output_len = (padded_len - self.kernel_size) / self.stride + 1;

        // Pad input
        let padded: Vec<Vec<f64>> = input
            .iter()
            .map(|ch| {
                let mut p = vec![0.0; self.padding];
                p.extend(ch);
                p.extend(vec![0.0; self.padding]);
                p
            })
            .collect();

        let mut output = vec![vec![0.0; output_len]; self.out_channels];
        for oc in 0..self.out_channels {
            for o in 0..output_len {
                let start = o * self.stride;
                let mut sum = self.biases[oc];
                for ic in 0..self.in_channels {
                    for k in 0..self.kernel_size {
                        sum += self.weights[oc][ic][k] * padded[ic][start + k];
                    }
                }
                output[oc][o] = sum.max(0.0); // ReLU
            }
        }
        output
    }
}

// ---------------------------------------------------------------------------
// 2-D Convolutional Layer
// ---------------------------------------------------------------------------

/// 2-D convolutional layer.
#[derive(Debug, Clone)]
pub struct Conv2D {
    /// Kernel weights: (out_channels, in_channels, kh, kw).
    pub weights: Array4<f64>,
    /// Bias per output channel.
    pub biases: Vec<f64>,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub padding_h: usize,
    pub padding_w: usize,
    pub out_channels: usize,
    pub in_channels: usize,
}

impl Conv2D {
    /// Create a new Conv2D layer with random Xavier initialisation.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let fan_in = in_channels * kernel_h * kernel_w;
        let scale = (2.0 / fan_in as f64).sqrt();
        let mut weights = Array4::zeros((out_channels, in_channels, kernel_h, kernel_w));
        for w in weights.iter_mut() {
            *w = rng.gen_range(-scale..scale);
        }
        let biases = vec![0.0; out_channels];
        Self {
            weights,
            biases,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            out_channels,
            in_channels,
        }
    }

    /// Forward pass.  Input shape: (batch, in_channels, H, W).
    /// Output shape: (batch, out_channels, H', W') with ReLU activation.
    pub fn forward(&self, input: &Array4<f64>) -> Array4<f64> {
        let batch = input.shape()[0];
        let _ic = input.shape()[1];
        let h = input.shape()[2];
        let w = input.shape()[3];

        let out_h = (h + 2 * self.padding_h - self.kernel_h) / self.stride_h + 1;
        let out_w = (w + 2 * self.padding_w - self.kernel_w) / self.stride_w + 1;

        let mut output = Array4::zeros((batch, self.out_channels, out_h, out_w));

        for b in 0..batch {
            for oc in 0..self.out_channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = self.biases[oc];
                        for ic in 0..self.in_channels {
                            for kh in 0..self.kernel_h {
                                for kw in 0..self.kernel_w {
                                    let ih = oh * self.stride_h + kh;
                                    let iw = ow * self.stride_w + kw;
                                    let ih = ih as isize - self.padding_h as isize;
                                    let iw = iw as isize - self.padding_w as isize;
                                    if ih >= 0
                                        && ih < h as isize
                                        && iw >= 0
                                        && iw < w as isize
                                    {
                                        sum += self.weights[[oc, ic, kh, kw]]
                                            * input[[b, ic, ih as usize, iw as usize]];
                                    }
                                }
                            }
                        }
                        output[[b, oc, oh, ow]] = sum.max(0.0); // ReLU
                    }
                }
            }
        }
        output
    }
}

// ---------------------------------------------------------------------------
// Max Pooling 2-D
// ---------------------------------------------------------------------------

/// 2-D max pooling layer.
#[derive(Debug, Clone)]
pub struct MaxPool2D {
    pub pool_h: usize,
    pub pool_w: usize,
}

impl MaxPool2D {
    pub fn new(pool_h: usize, pool_w: usize) -> Self {
        Self { pool_h, pool_w }
    }

    /// Forward pass.  Input: (batch, channels, H, W) -> (batch, channels, H/ph, W/pw).
    pub fn forward(&self, input: &Array4<f64>) -> Array4<f64> {
        let batch = input.shape()[0];
        let channels = input.shape()[1];
        let h = input.shape()[2];
        let w = input.shape()[3];
        let out_h = h / self.pool_h;
        let out_w = w / self.pool_w;

        let mut output = Array4::from_elem((batch, channels, out_h, out_w), f64::NEG_INFINITY);

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        for ph in 0..self.pool_h {
                            for pw in 0..self.pool_w {
                                let ih = oh * self.pool_h + ph;
                                let iw = ow * self.pool_w + pw;
                                if ih < h && iw < w {
                                    let val = input[[b, c, ih, iw]];
                                    if val > output[[b, c, oh, ow]] {
                                        output[[b, c, oh, ow]] = val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        output
    }
}

// ---------------------------------------------------------------------------
// Fully-Connected (Dense) Layer
// ---------------------------------------------------------------------------

/// A dense (fully-connected) layer.
#[derive(Debug, Clone)]
pub struct Dense {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub use_relu: bool,
}

impl Dense {
    pub fn new(in_features: usize, out_features: usize, use_relu: bool) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / in_features as f64).sqrt();
        let mut weights = Array2::zeros((out_features, in_features));
        for w in weights.iter_mut() {
            *w = rng.gen_range(-scale..scale);
        }
        let biases = Array1::zeros(out_features);
        Self {
            weights,
            biases,
            use_relu,
        }
    }

    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut output = self.weights.dot(input) + &self.biases;
        if self.use_relu {
            output.mapv_inplace(|x| x.max(0.0));
        }
        output
    }
}

// ---------------------------------------------------------------------------
// Softmax utility
// ---------------------------------------------------------------------------

/// Compute softmax of a 1-D array.
pub fn softmax(logits: &Array1<f64>) -> Array1<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps = logits.mapv(|x| (x - max_val).exp());
    let sum = exps.sum();
    exps / sum
}

// ---------------------------------------------------------------------------
// Classification head: direction prediction
// ---------------------------------------------------------------------------

/// Prediction direction for mid-price movement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direction {
    Up,
    Down,
    Stable,
}

impl std::fmt::Display for Direction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Direction::Up => write!(f, "Up"),
            Direction::Down => write!(f, "Down"),
            Direction::Stable => write!(f, "Stable"),
        }
    }
}

// ---------------------------------------------------------------------------
// CNN Model (simplified DeepLOB-style)
// ---------------------------------------------------------------------------

/// A simplified CNN model for LOB mid-price direction prediction.
pub struct CnnLobModel {
    pub conv1: Conv2D,
    pub conv2: Conv2D,
    pub pool: MaxPool2D,
    pub fc1: Dense,
    pub fc2: Dense,
    /// Expected spatial size after conv+pool for the flatten step.
    pub flat_size: usize,
}

impl CnnLobModel {
    /// Build a new model for the given LOB image dimensions.
    /// `levels`: number of price levels per side.
    /// `time_steps`: number of time-step columns.
    pub fn new(levels: usize, time_steps: usize) -> Self {
        let h = 2 * levels; // input height
        let w = time_steps; // input width

        // Conv1: (1, 32, kh=3, kw=3), padding=1 => output (32, h, w)
        let conv1 = Conv2D::new(1, 32, 3, 3, 1, 1, 1, 1);
        // Conv2: (32, 32, kh=3, kw=3), padding=1 => output (32, h, w)
        let conv2 = Conv2D::new(32, 32, 3, 3, 1, 1, 1, 1);
        // Pool 2×2 => output (32, h/2, w/2)
        let pool = MaxPool2D::new(2, 2);

        let pooled_h = h / 2;
        let pooled_w = w / 2;
        let flat_size = 32 * pooled_h * pooled_w;

        let fc1 = Dense::new(flat_size, 128, true);
        let fc2 = Dense::new(128, 3, false);

        Self {
            conv1,
            conv2,
            pool,
            fc1,
            fc2,
            flat_size,
        }
    }

    /// Run forward inference on an LOB image tensor.
    /// Returns (direction, probabilities).
    pub fn predict(&self, input: &Array4<f64>) -> (Direction, Array1<f64>) {
        let x = self.conv1.forward(input);
        let x = self.conv2.forward(&x);
        let x = self.pool.forward(&x);

        // Flatten
        let flat: Vec<f64> = x.iter().cloned().collect();
        let flat = Array1::from_vec(flat);
        assert_eq!(
            flat.len(),
            self.flat_size,
            "Flat size mismatch: got {} expected {}",
            flat.len(),
            self.flat_size
        );

        let h = self.fc1.forward(&flat);
        let logits = self.fc2.forward(&h);
        let probs = softmax(&logits);

        let dir = if probs[0] >= probs[1] && probs[0] >= probs[2] {
            Direction::Up
        } else if probs[1] >= probs[0] && probs[1] >= probs[2] {
            Direction::Down
        } else {
            Direction::Stable
        };

        (dir, probs)
    }
}

// ---------------------------------------------------------------------------
// Bybit orderbook fetching
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct BybitResponse {
    result: BybitOrderbook,
}

#[derive(Debug, Deserialize)]
struct BybitOrderbook {
    /// Ask levels: [[price, qty], ...]
    a: Vec<Vec<String>>,
    /// Bid levels: [[price, qty], ...]
    b: Vec<Vec<String>>,
}

/// A single orderbook snapshot from Bybit.
#[derive(Debug, Clone)]
pub struct OrderbookSnapshot {
    /// Ask volumes sorted by price ascending (best ask first).
    pub ask_volumes: Vec<f64>,
    /// Bid volumes sorted by price descending (best bid first).
    pub bid_volumes: Vec<f64>,
    /// Best ask price.
    pub best_ask: f64,
    /// Best bid price.
    pub best_bid: f64,
}

impl OrderbookSnapshot {
    /// Compute mid price.
    pub fn mid_price(&self) -> f64 {
        (self.best_ask + self.best_bid) / 2.0
    }
}

/// Fetch a live orderbook snapshot from Bybit for the given symbol.
/// `levels` controls how many levels on each side to return.
pub fn fetch_bybit_orderbook(
    symbol: &str,
    levels: usize,
) -> anyhow::Result<OrderbookSnapshot> {
    let url = format!(
        "https://api.bybit.com/v5/market/orderbook?category=spot&symbol={}&limit={}",
        symbol, levels
    );
    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client.get(&url).send()?.json()?;

    let ask_volumes: Vec<f64> = resp
        .result
        .a
        .iter()
        .take(levels)
        .map(|entry| entry[1].parse::<f64>().unwrap_or(0.0))
        .collect();
    let bid_volumes: Vec<f64> = resp
        .result
        .b
        .iter()
        .take(levels)
        .map(|entry| entry[1].parse::<f64>().unwrap_or(0.0))
        .collect();

    let best_ask = resp
        .result
        .a
        .first()
        .map(|e| e[0].parse::<f64>().unwrap_or(0.0))
        .unwrap_or(0.0);
    let best_bid = resp
        .result
        .b
        .first()
        .map(|e| e[0].parse::<f64>().unwrap_or(0.0))
        .unwrap_or(0.0);

    Ok(OrderbookSnapshot {
        ask_volumes,
        bid_volumes,
        best_ask,
        best_bid,
    })
}

/// Build an LOB image from a series of orderbook snapshots.
pub fn build_lob_image(snapshots: &[OrderbookSnapshot], levels: usize) -> LobImage {
    let time_steps = snapshots.len();
    let mut img = LobImage::new(levels, time_steps);
    for (t, snap) in snapshots.iter().enumerate() {
        let asks: Vec<f64> = snap
            .ask_volumes
            .iter()
            .copied()
            .chain(std::iter::repeat(0.0))
            .take(levels)
            .collect();
        let bids: Vec<f64> = snap
            .bid_volumes
            .iter()
            .copied()
            .chain(std::iter::repeat(0.0))
            .take(levels)
            .collect();
        img.set_snapshot(t, &asks, &bids);
    }
    img.normalize();
    img
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lob_image_creation() {
        let levels = 5;
        let time_steps = 4;
        let mut img = LobImage::new(levels, time_steps);
        assert_eq!(img.data.shape(), &[10, 4]);

        let asks = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let bids = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        img.set_snapshot(0, &asks, &bids);
        assert_eq!(img.data[[0, 0]], 1.0);
        assert_eq!(img.data[[5, 0]], 5.0);
    }

    #[test]
    fn test_lob_normalize() {
        let mut img = LobImage::new(3, 2);
        img.set_snapshot(0, &[10.0, 20.0, 30.0], &[30.0, 20.0, 10.0]);
        img.set_snapshot(1, &[5.0, 15.0, 25.0], &[25.0, 15.0, 5.0]);
        img.normalize();

        // After z-score per column, mean of each column should be ~0
        let col0: Vec<f64> = (0..6).map(|i| img.data[[i, 0]]).collect();
        let mean: f64 = col0.iter().sum::<f64>() / col0.len() as f64;
        assert!(mean.abs() < 1e-6, "mean should be ~0, got {}", mean);
    }

    #[test]
    fn test_conv1d_output_shape() {
        let conv = Conv1D::new(1, 4, 3, 1, 1);
        let input = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let output = conv.forward(&input);
        assert_eq!(output.len(), 4); // 4 output channels
        assert_eq!(output[0].len(), 5); // same length with padding=1
    }

    #[test]
    fn test_conv2d_output_shape() {
        let conv = Conv2D::new(1, 8, 3, 3, 1, 1, 1, 1);
        let input = Array4::zeros((1, 1, 10, 6));
        let output = conv.forward(&input);
        assert_eq!(output.shape(), &[1, 8, 10, 6]);
    }

    #[test]
    fn test_max_pool() {
        let pool = MaxPool2D::new(2, 2);
        let mut input = Array4::zeros((1, 1, 4, 4));
        input[[0, 0, 0, 0]] = 1.0;
        input[[0, 0, 0, 1]] = 3.0;
        input[[0, 0, 1, 0]] = 2.0;
        input[[0, 0, 1, 1]] = 4.0;
        let output = pool.forward(&input);
        assert_eq!(output.shape(), &[1, 1, 2, 2]);
        assert_eq!(output[[0, 0, 0, 0]], 4.0);
    }

    #[test]
    fn test_softmax() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let probs = softmax(&logits);
        let sum: f64 = probs.sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_dense_layer() {
        let dense = Dense::new(4, 3, true);
        let input = Array1::from_vec(vec![1.0, -1.0, 0.5, 2.0]);
        let output = dense.forward(&input);
        assert_eq!(output.len(), 3);
        // With ReLU, all outputs should be >= 0
        for &v in output.iter() {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_cnn_model_predict() {
        let levels = 10;
        let time_steps = 20;
        let model = CnnLobModel::new(levels, time_steps);
        let img = LobImage::new(levels, time_steps);
        let tensor = img.to_tensor();
        let (dir, probs) = model.predict(&tensor);

        // Probabilities should sum to 1
        let sum: f64 = probs.sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Direction should be one of the three
        assert!(
            dir == Direction::Up || dir == Direction::Down || dir == Direction::Stable
        );
    }

    #[test]
    fn test_build_lob_image_from_snapshots() {
        let snaps = vec![
            OrderbookSnapshot {
                ask_volumes: vec![10.0, 20.0, 30.0],
                bid_volumes: vec![30.0, 20.0, 10.0],
                best_ask: 100.0,
                best_bid: 99.0,
            },
            OrderbookSnapshot {
                ask_volumes: vec![15.0, 25.0, 35.0],
                bid_volumes: vec![35.0, 25.0, 15.0],
                best_ask: 101.0,
                best_bid: 100.0,
            },
        ];
        let img = build_lob_image(&snaps, 3);
        assert_eq!(img.data.shape(), &[6, 2]);
        // After normalization the values should be centred around 0
        let mean = img.data.sum() / img.data.len() as f64;
        assert!(mean.abs() < 1e-6);
    }
}
