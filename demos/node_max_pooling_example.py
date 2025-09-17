import torch




if __name__ == "__main__":
    node_features = torch.randn((6, 16))  # 6 nodes, each with 16 features
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 2],
                               [1, 0, 3, 2, 5, 4, 2, 0]])  # Example edge connections
    edge_weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Example edge weights