# CNN LOB Prediction — Explained Simply

## What is this about?

Imagine recognizing patterns in a market chart the same way you recognize faces in photos. That is exactly what we do here! We take a picture of the order book — all the buy and sell orders lined up at different prices — and let a special kind of AI called a Convolutional Neural Network (CNN) look at that picture to predict where the price will go next.

## What is a Limit Order Book?

Think of a market like a big board at a farmers' market. On one side, buyers write down how much they want to pay and how many apples they want. On the other side, sellers write down their prices and quantities. This board is the "order book."

Now imagine taking a photo of that board every second. If you stack all those photos side by side, you get something like a movie — you can see how the board changes over time. That movie is what we call an "LOB image."

## What is a CNN?

A CNN is like a detective with a magnifying glass. Instead of looking at the whole picture at once, it slides a small window (called a "filter") across the image, looking for tiny patterns:

- **Pattern 1**: "There are way more buyers than sellers at this price!" (demand imbalance)
- **Pattern 2**: "The sellers keep disappearing over time!" (fading supply)
- **Pattern 3**: "A big wall of orders just appeared!" (support or resistance)

The CNN learns which patterns matter most for predicting if the price will go **up**, **down**, or **stay the same**.

## How does it work step by step?

1. **Take snapshots**: We ask the exchange (Bybit) for the current order book every moment.
2. **Build a picture**: We stack these snapshots into a grid — prices go top to bottom, time goes left to right.
3. **Slide the magnifying glass**: The CNN slides its filters across the picture, finding patterns.
4. **Shrink the picture**: We use "pooling" to make the picture smaller while keeping the most important info (like making a thumbnail).
5. **Make a decision**: A final layer looks at all the patterns found and says: "I think the price will go UP with 70% confidence!"

## A real-world analogy

Think of it like weather forecasting with satellite images. Meteorologists look at a series of cloud photos to predict rain. Similarly, our CNN looks at a series of order book snapshots to predict price movement. The CNN is just much faster and can spot patterns too subtle for human eyes!

## Why Rust?

We write the code in Rust because it is fast — really fast. When you are trying to predict prices that change in milliseconds, every bit of speed counts. Rust helps us make predictions before the market moves on.

## Key ideas to remember

- The **order book** is like a scoreboard of all buy and sell orders
- We turn it into a **picture** by stacking snapshots over time
- A **CNN** slides small filters across that picture to find patterns
- It predicts whether the price will go **up**, **down**, or **stay flat**
- This is similar to how your phone recognizes faces — but for market data!
