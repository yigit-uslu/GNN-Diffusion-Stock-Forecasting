#!/usr/bin/env python3

def test_original_issue():
    """Test the original floating-point precision issue"""
    pool_ratio = 0.8
    graph_pooling_factor = 1 / pool_ratio
    print(f"Original calculation: 1 / {pool_ratio} = {graph_pooling_factor}")
    print(f"Is exactly 1.25? {graph_pooling_factor == 1.25}")
    
def test_fraction_fix():
    """Test using fractions to fix the issue"""
    from fractions import Fraction
    pool_ratio = 0.8
    graph_pooling_factor = float(Fraction(1) / Fraction(pool_ratio).limit_denominator())
    print(f"Fraction calculation: 1 / {pool_ratio} = {graph_pooling_factor}")
    print(f"Is exactly 1.25? {graph_pooling_factor == 1.25}")

def test_exact_fix():
    """Test using exact values for common ratios"""
    pool_ratio = 0.8
    if abs(pool_ratio - 0.8) < 1e-10:
        graph_pooling_factor = 1.25  # Exactly 5/4
    else:
        graph_pooling_factor = 1 / pool_ratio
    print(f"Exact fix calculation: pool_ratio={pool_ratio}, factor={graph_pooling_factor}")
    print(f"Is exactly 1.25? {graph_pooling_factor == 1.25}")

def test_downsampling_sequence():
    """Test the downsampling sequence to see where it goes wrong"""
    print("\n=== Testing downsampling sequence ===")
    
    # Test original problem
    print("Original problematic sequence:")
    nodes = 100
    pool_ratio = 0.8
    graph_pooling_factor = 1 / pool_ratio
    print(f"Initial: graph_pooling_factor = 1 / {pool_ratio} = {graph_pooling_factor}")
    
    for depth in range(3):
        new_nodes = int(nodes / graph_pooling_factor)
        actual_factor = nodes / new_nodes
        print(f"Depth {depth+1}: {nodes} -> {new_nodes}, actual_factor = {actual_factor}")
        nodes = new_nodes
    
    print("\nFixed sequence:")
    nodes = 100
    graph_pooling_factor = 1.25  # Exactly
    print(f"Initial: graph_pooling_factor = {graph_pooling_factor}")
    
    for depth in range(3):
        new_nodes = int(nodes / graph_pooling_factor)
        actual_factor = nodes / new_nodes
        print(f"Depth {depth+1}: {nodes} -> {new_nodes}, actual_factor = {actual_factor}")
        nodes = new_nodes

if __name__ == "__main__":
    test_original_issue()
    print()
    test_fraction_fix()
    print()
    test_exact_fix()
    print()
    test_downsampling_sequence()
