import torch
import numpy as np
import matplotlib.pyplot as plt

def compare_covariance_vs_correlation():
    """
    Demonstrate the difference between covariance and correlation methods
    for creating stock graphs.
    """
    
    # Create synthetic stock data with different price levels but similar patterns
    torch.manual_seed(42)
    n_days = 100
    
    # Stock A: Low price (~$10), high volatility
    base_return_A = torch.randn(n_days) * 0.03  # 3% daily volatility
    price_A = 10 * torch.cumprod(1 + base_return_A, dim=0)
    
    # Stock B: High price (~$200), low volatility  
    base_return_B = torch.randn(n_days) * 0.01  # 1% daily volatility
    price_B = 200 * torch.cumprod(1 + base_return_B, dim=0)
    
    # Stock C: Medium price (~$50), correlated with A
    base_return_C = 0.8 * base_return_A + 0.6 * torch.randn(n_days) * 0.02
    price_C = 50 * torch.cumprod(1 + base_return_C, dim=0)
    
    # Stock D: High price (~$300), correlated with B
    base_return_D = 0.7 * base_return_B + 0.714 * torch.randn(n_days) * 0.015
    price_D = 300 * torch.cumprod(1 + base_return_D, dim=0)
    
    # Combine into price matrix
    prices = torch.stack([price_A, price_B, price_C, price_D])
    stock_names = ['Stock A ($10)', 'Stock B ($200)', 'Stock C ($50)', 'Stock D ($300)']
    
    print("Stock Price Summary:")
    print("-" * 50)
    for i, name in enumerate(stock_names):
        print(f"{name}: Mean=${prices[i].mean():.2f}, Std=${prices[i].std():.2f}")
    
    # Compute covariance and correlation matrices
    cov_matrix = torch.cov(prices)
    corr_matrix = torch.corrcoef(prices)
    
    print(f"\nCovariance Matrix:")
    print("-" * 30)
    print(cov_matrix.numpy())
    
    print(f"\nCorrelation Matrix:")
    print("-" * 30)
    print(corr_matrix.numpy())
    
    # Key observations
    print(f"\nKey Observations:")
    print("-" * 50)
    
    # Compare A-C relationship (designed to be correlated)
    cov_AC = cov_matrix[0, 2].item()
    corr_AC = corr_matrix[0, 2].item()
    print(f"Stock A-C: Covariance={cov_AC:.3f}, Correlation={corr_AC:.3f}")
    
    # Compare B-D relationship (designed to be correlated)  
    cov_BD = cov_matrix[1, 3].item()
    corr_BD = corr_matrix[1, 3].item()
    print(f"Stock B-D: Covariance={cov_BD:.3f}, Correlation={corr_BD:.3f}")
    
    # Compare A-B relationship (should be weak)
    cov_AB = cov_matrix[0, 1].item()
    corr_AB = corr_matrix[0, 1].item()
    print(f"Stock A-B: Covariance={cov_AB:.3f}, Correlation={corr_AB:.3f}")
    
    print(f"\nGraph Implications:")
    print("-" * 50)
    
    # Simulate thresholding
    cov_threshold = 5.0
    corr_threshold = 0.3
    
    # Count edges under each method
    cov_edges = (torch.abs(cov_matrix) > cov_threshold).sum() - 4  # subtract diagonal
    corr_edges = (torch.abs(corr_matrix) > corr_threshold).sum() - 4
    
    print(f"Covariance method (threshold={cov_threshold}): {cov_edges} edges")
    print(f"Correlation method (threshold={corr_threshold}): {corr_edges} edges")
    
    # Show which connections would be made
    print(f"\nConnections made by each method:")
    print("-" * 40)
    
    print("Covariance connections:")
    for i in range(4):
        for j in range(i+1, 4):
            if torch.abs(cov_matrix[i, j]) > cov_threshold:
                print(f"  {stock_names[i]} <-> {stock_names[j]} (cov={cov_matrix[i,j]:.2f})")
    
    print("Correlation connections:")  
    for i in range(4):
        for j in range(i+1, 4):
            if torch.abs(corr_matrix[i, j]) > corr_threshold:
                print(f"  {stock_names[i]} <-> {stock_names[j]} (corr={corr_matrix[i,j]:.3f})")


def theoretical_differences():
    """
    Explain the theoretical differences between covariance and correlation.
    """
    print("\n" + "="*70)
    print("THEORETICAL DIFFERENCES")
    print("="*70)
    
    print("""
1. MATHEMATICAL DEFINITION:
   
   Covariance: Cov(X,Y) = E[(X - μₓ)(Y - μᵧ)]
   - Measures average product of deviations from means
   - Units: [Price₁] × [Price₂]
   - Range: (-∞, +∞)
   
   Correlation: Corr(X,Y) = Cov(X,Y) / (σₓ × σᵧ)
   - Normalized covariance by standard deviations
   - Unitless
   - Range: [-1, +1]

2. PRACTICAL IMPLICATIONS FOR STOCK GRAPHS:

   a) Scale Effects:
      - Covariance: High-priced stocks dominate the graph structure
      - Correlation: All stocks treated equally regardless of price level
   
   b) Economic Interpretation:
      - Covariance: "How much do prices move together in dollar terms?"
      - Correlation: "How synchronized are the price movements?"
   
   c) Graph Connectivity:
      - Covariance: May miss important relationships between different price tiers
      - Correlation: Better captures fundamental business relationships
   
   d) Threshold Selection:
      - Covariance: Threshold depends on price scales in your dataset
      - Correlation: Threshold has universal meaning (0.7 = strong relationship)

3. WHEN TO USE EACH:

   Use COVARIANCE when:
   - You care about absolute dollar risk/exposure
   - Building portfolio optimization models
   - All stocks are in similar price ranges
   - You want to weight by market cap/price level
   
   Use CORRELATION when:
   - You want to capture pure co-movement patterns
   - Stocks have very different price levels
   - Building relationship/similarity graphs
   - You want sector/industry clustering (recommended)
   
4. RECOMMENDATION FOR STOCK FORECASTING:
   
   Generally prefer CORRELATION because:
   - Better captures fundamental business relationships
   - More robust to different price scales
   - Easier to interpret and set thresholds
   - More commonly used in financial literature
   """)


if __name__ == "__main__":
    compare_covariance_vs_correlation()
    theoretical_differences()
