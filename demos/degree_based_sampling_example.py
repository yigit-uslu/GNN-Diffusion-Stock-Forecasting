import numpy as np
import torch 

def seed_everything(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def degree_based_sampling_example(N = 16, pool_ratio = 0.75):

    # Generate a random graph with N nodes
    graph = np.random.rand(N, N)
    graph = (graph + graph.T) / 2  # Make it symmetric
    np.fill_diagonal(graph, 0)  # No self-loops

    graph = (graph > 0.5).astype(int)  # Sparsify the graph
    graph = torch.from_numpy(graph)

    # Compute node degrees
    degrees = torch.sum(graph, dim=1)

    orig_degrees = degrees.clone()
    orig_N = N

    print("Node degrees: ", degrees.tolist())
    selections = torch.arange(N, dtype=torch.float32)
    selection_matrix = torch.eye(N, dtype=torch.float32)

    all_selection_matrices = []
    for depth in range(2):
        print("\n*****************************************************\n"
              + f"Pooling at depth {depth + 1} with pool ratio = {pool_ratio}\n" + 
              "*****************************************************\n")

        # Create a pool of nodes based on the degree
        num_pooled_nodes = int(N * pool_ratio)
        _, indices = torch.topk(degrees, num_pooled_nodes,
                                        largest = True, sorted = False)
    
        print(f"Selected node indices (top {num_pooled_nodes} by degree): {indices.tolist()}")

        sorted_new_indices = torch.sort(indices).values
        degrees = degrees[sorted_new_indices]
        print(f"Sorted selected node indices: {sorted_new_indices.tolist()}")

        # Create a matrix that selects the pooled nodes among the original nodes
        new_selection_matrix = torch.zeros((num_pooled_nodes, N))
        for i, idx in enumerate(sorted_new_indices):
            new_selection_matrix[i, idx] = 1.0

        print("Selections: ", new_selection_matrix @ torch.arange(N, dtype = torch.float32).view(-1,))
        print("Expected: ", sorted_new_indices.float())
        # print("Original degrees of selected nodes: ", orig_degrees[sorted_new_indices].float())
        assert torch.allclose(new_selection_matrix @ torch.arange(N, dtype=torch.float32).view(-1,),
                            sorted_new_indices.float()), "Selection matrix is incorrect."
        

        selection_matrix = new_selection_matrix @ selection_matrix # cumulative selections
        print("Cumulative selections: ", selection_matrix @ torch.arange(orig_N, dtype=torch.float32).view(-1,))
        # print("Cumulative expected: ", sorted_indices[sorted_new_indices].float())

        # print("Original degrees of cumulatively selected nodes: ", orig_degrees[sorted_indices[sorted_new_indices]].float())
    

        print("\n*****************************************************\n"
            + f"Repeating the pooling to verify consistency...\n" + 
            "*****************************************************\n")

        all_selection_matrices.append(selection_matrix)
        N = num_pooled_nodes


    
    # ### Repeat the pooling ###
    # num_pooled_nodes = int(N * pool_ratio)
    # new_pool_values, new_indices = torch.topk(pool_values, num_pooled_nodes,
    #                                           largest = True, sorted = False)
    
    # sorted_new_indices = torch.sort(new_indices).values
    # print(f"Sorted selected node indices after second pooling: {sorted_new_indices.tolist()}")
    
    # # Create a matrix that selects the pooled nodes among the original nodes
    # new_selection_matrix = torch.zeros((num_pooled_nodes, N))
    # for i, idx in enumerate(sorted_new_indices):
    #     new_selection_matrix[i, idx] = 1.0

    # print("Selections: ", new_selection_matrix @ torch.arange(N, dtype=torch.float32).view(-1,))
    # print("Expected: ", sorted_new_indices.float())
    # print("Original degrees of selected nodes: ", pool_values[sorted_new_indices].float())
    # assert torch.allclose(new_selection_matrix @ torch.arange(N, dtype=torch.float32).view(-1,),
    #                        sorted_new_indices.float()), "Selection matrix is incorrect."
    

    # print("Cumulative selections: ", new_selection_matrix @ selection_matrix @ torch.arange(orig_N, dtype=torch.float32).view(-1,))
    # print("Cumulative expected: ", sorted_indices[sorted_new_indices].float())

    # print("Original degrees of cumulatively selected nodes: ", orig_degrees[sorted_indices[sorted_new_indices]].float())

    return 0




if __name__ == "__main__":

    seed_everything(42)
    degree_based_sampling_example()