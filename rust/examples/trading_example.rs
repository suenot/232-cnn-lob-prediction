//! Trading example: Fetch BTCUSDT orderbook from Bybit, build LOB image, run CNN prediction.

use cnn_lob_prediction::{
    build_lob_image, fetch_bybit_orderbook, CnnLobModel, OrderbookSnapshot,
};
use std::thread;
use std::time::Duration;

fn main() -> anyhow::Result<()> {
    let symbol = "BTCUSDT";
    let levels = 10;
    let num_snapshots = 20;

    println!("=== CNN LOB Prediction — Trading Example ===");
    println!("Symbol:     {}", symbol);
    println!("Levels:     {} per side", levels);
    println!("Snapshots:  {}", num_snapshots);
    println!();

    // ------------------------------------------------------------------
    // Step 1: Collect orderbook snapshots
    // ------------------------------------------------------------------
    println!("Fetching {} orderbook snapshots from Bybit...", num_snapshots);
    let mut snapshots: Vec<OrderbookSnapshot> = Vec::new();

    for i in 0..num_snapshots {
        match fetch_bybit_orderbook(symbol, levels) {
            Ok(snap) => {
                println!(
                    "  Snapshot {}/{}: mid_price = {:.2}  best_bid = {:.2}  best_ask = {:.2}",
                    i + 1,
                    num_snapshots,
                    snap.mid_price(),
                    snap.best_bid,
                    snap.best_ask,
                );
                snapshots.push(snap);
            }
            Err(e) => {
                eprintln!("  Warning: failed to fetch snapshot {}: {}", i + 1, e);
                // On error, duplicate the last snapshot to keep the time series intact
                if let Some(last) = snapshots.last() {
                    snapshots.push(last.clone());
                }
            }
        }
        if i < num_snapshots - 1 {
            thread::sleep(Duration::from_millis(500));
        }
    }

    if snapshots.len() < 2 {
        anyhow::bail!("Not enough snapshots collected (need at least 2).");
    }

    // ------------------------------------------------------------------
    // Step 2: Build LOB image
    // ------------------------------------------------------------------
    println!("\nBuilding LOB image ({} levels × {} time steps)...", 2 * levels, snapshots.len());
    let lob_image = build_lob_image(&snapshots, levels);
    let tensor = lob_image.to_tensor();
    println!("  Tensor shape: {:?}", tensor.shape());

    // ------------------------------------------------------------------
    // Step 3: Run CNN model
    // ------------------------------------------------------------------
    println!("\nInitialising CNN model (random weights — demo only)...");
    let model = CnnLobModel::new(levels, snapshots.len());
    let (direction, probs) = model.predict(&tensor);

    println!("\n=== Prediction Results ===");
    println!("  P(Up)     = {:.4}", probs[0]);
    println!("  P(Down)   = {:.4}", probs[1]);
    println!("  P(Stable) = {:.4}", probs[2]);
    println!("  Direction = {}", direction);

    // ------------------------------------------------------------------
    // Step 4: Simple signal interpretation
    // ------------------------------------------------------------------
    let first_mid = snapshots.first().unwrap().mid_price();
    let last_mid = snapshots.last().unwrap().mid_price();
    let actual_change_bps = (last_mid - first_mid) / first_mid * 10_000.0;

    println!("\n=== Market Context ===");
    println!("  First mid price: {:.2}", first_mid);
    println!("  Last mid price:  {:.2}", last_mid);
    println!("  Change:          {:.2} bps", actual_change_bps);
    println!();
    println!("Note: This model uses random weights and is for demonstration only.");
    println!("      For real trading, train the model on historical LOB data.");

    Ok(())
}
