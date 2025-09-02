import torch
from torch_geometric.loader import DataLoader
from datasets.SP100Stocks import StocksDataDiffusion

def test_info_batching():
    """Test that info dictionary is preserved correctly during batching"""
    
    # Create sample data objects with info dictionaries matching your actual structure
    info1 = {
        'Features': ['feature1', 'feature2', 'feature3'],
        'Target': 'NormClose', 
        'Num_nodes': 100
    }
    
    info2 = {
        'Features': ['different_feature1', 'different_feature2'],
        'Target': 'NormClose',
        'Num_nodes': 100
    }
    
    # Create sample tensors
    x = torch.randn(10, 5, 3)  # 10 nodes, 5 features, 3 time steps
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    edge_weight = torch.ones(3)
    close_price = torch.randn(10, 3)
    y = torch.randn(10, 1)
    close_price_y = torch.randn(10, 1)
    timestamp = torch.tensor([0])
    stocks_index = torch.arange(10)
    
    # Create two data objects
    data1 = StocksDataDiffusion(
        x=x, edge_index=edge_index, edge_weight=edge_weight,
        close_price=close_price, y=y, close_price_y=close_price_y,
        timestamp=timestamp, stocks_index=stocks_index, info=info1
    )
    
    data2 = StocksDataDiffusion(
        x=x, edge_index=edge_index, edge_weight=edge_weight,
        close_price=close_price, y=y, close_price_y=close_price_y,
        timestamp=timestamp, stocks_index=stocks_index, info=info2
    )
    
    print("Before batching:")
    # print(f"Data1 info: {data1.info}")
    # print(f"Data2 info: {data2.info}")
    
    # Create a dataloader to batch the data
    dataset = [data1, data2]
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # Get the batched data
    for batch in loader:
        print("After batching:")
        print(f"Batch has info: {hasattr(batch, 'info')}")
        
        if hasattr(batch, 'info'):
            print(f"Batch info: {batch.info}")
        else:
            print("Batch info: None")
        
        print(f"Batch x shape: {batch.x.shape}")
        print(f"Batch timestamp: {batch.timestamp}")
        print(f"Batch stocks_index: {batch.stocks_index}")
        
        # Check if any info keys were accidentally batched as separate attributes
        info_keys = ['Features', 'Target', 'Num_nodes']
        for key in info_keys:
            if hasattr(batch, key):
                print(f"WARNING: {key} found as separate attribute: {getattr(batch, key)}")
            else:
                print(f"Good: {key} not found as separate attribute")
        
        break

if __name__ == "__main__":
    test_info_batching()
