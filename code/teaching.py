import numpy as np
import random
from scipy.optimize import linprog

from helper import get_unique_normalized_vectors, is_equal_vectors, process_text, l2_norm
from instance_generator import get_feature_differences

# find supporting vectors from a bunch of unit vectors that define a polyhedral cone
def get_supporting_vectors(X, verbose=0):

    # child function to solve the linear program
    # input :   
    #   1. X - a n x d dimensional array
    # output :
    #   1. S - a m x d dimensional array, where each row is a supporting vector of the polyhedral cone defined by X


    # linear program to 
    def solver_linear_program(A, v):

        # input :   
        #   1. A - a n x d dimensional array
        #   2. v - a d dimensional array
        b = np.ones(A.shape[0])

        bounds = [(None, None) for i in range(len(v))]

        if(verbose>=2):
            print("Constraint matrix : ", )
            print(process_text(A))
            print("Cost vector : ", )
            print(process_text(v))
            print()
        
        result = linprog(v, A_ub=-A, b_ub=-b, bounds=bounds, method='highs')

        if(verbose>=2):
            print("Optimal value : ", result.fun)
            print("Optimal solution : ", result.x)
            print()

        return result

    n, d = X.shape
    S = X[0,:]
    
    # iteratively check if i^th point is in the cone of remaining points
    for i in range(1, n):

        A = np.vstack((S, X[i+1:]))  # A is the current surviving set of extreme rays 
        
        
        # testing if the i^th point is in the cone of the remaining vectors
        v = X[i,:]
        result = solver_linear_program(A, v)
        status, objective = result.status, result.fun

        # if the point is not in the cone of the remaining vectors, add it to the supporting vectors
        if(status==3):
            S = np.vstack((S, v))

    return S


def get_set_cover_instance(Psi_dic, hat_E, tol=1e-10):

        S = [key[0] for key in Psi_dic.keys()]
        V = {s: set() for s in S}
        U = set({i for i in range(hat_E.shape[0])})

        for key, value in Psi_dic.items():
            s = key[0]
            Psi_saa = value
            hat_Psi_saa = (Psi_saa)/l2_norm(Psi_saa)

            for i in range(hat_E.shape[0]):
                if is_equal_vectors(hat_E[i,:], hat_Psi_saa, tol=tol):
                    V[s].add(i)
        
        return U, V


def solve_greedy_set_cover(U, V):
    """
    Greedy algorithm to find a set cover.
    
    Parameters:
    - U (list of int): The list of all elements that need to be covered.
    - V (dict of state tuple to list of int): A dictionary where each key is a tuple that indexes the subset.
    
    Returns:
    - list of tuples: The tuple indices of subsets that form the set cover.
    """
    uncovered = set(U)  # Convert the list to a set for faster operations
    cover_indices = []

    # Continue until all elements are covered
    while uncovered:

        # iterate over the subsets and find one' that covers the most uncovered elements
        current_best_subset_idx = None
        current_best_subset_cover = None
        current_best_cover_size = -1

        for idx, subset in V.items():
            current_cover = uncovered.intersection(subset)  # number of uncovered elements subset is covering

            if len(current_cover) > current_best_cover_size:
                current_best_cover_size = len(current_cover)   
                current_best_subset_idx = idx
                current_best_subset_cover = current_cover

        if current_best_cover_size > 0:
            # Add the best subset to the solution and remove its elements from uncovered
            cover_indices.append(current_best_subset_idx)
            uncovered.difference_update(current_best_subset_cover)
            V.pop(current_best_subset_idx)

        else:
            raise ValueError("No subset can cover all elements.")

    return cover_indices

def find_optimal_teaching_set(S, A, Phi_dic, pi):

    Psi_dic = get_feature_differences(S, A, Phi_dic, pi)
    Psi = np.array([value for key, value in Psi_dic.items()])


    # compute supporting vectors
    unique_hat_Psi = get_unique_normalized_vectors(Psi)
    hat_E = get_supporting_vectors(unique_hat_Psi)

    U, V = get_set_cover_instance(Psi_dic, hat_E, tol=1e-10)

    T = solve_greedy_set_cover(U, V)
    return T, hat_E, Psi_dic


def get_random_set_cover_size(U, V, num_iterations):
    total_draws, all_drawn_subsets = 0, []

    for _ in range(num_iterations):
        uncovered = set(U)
        num_draws = 0
        selected_subsets, drawn_subsets = [], []

        while uncovered:
            num_draws += 1
            subset_index = random.choice(list(V.keys()))
            drawn_subsets.append(subset_index)
            subset = V[subset_index]
            if subset.intersection(uncovered):
                selected_subsets.append(subset_index)
                uncovered -= subset
        total_draws += num_draws
        all_drawn_subsets.append(drawn_subsets)

    avg_draws = total_draws / num_iterations
    return avg_draws, all_drawn_subsets

def find_random_teaching_set(Psi_dic, num_iterations, hat_E=None):

    if(hat_E is None):
        Psi = np.array([value for key, value in Psi_dic.items()])

        # compute supporting vectors
        unique_hat_Psi = get_unique_normalized_vectors(Psi)
        hat_E = get_supporting_vectors(unique_hat_Psi)

    # form and solve set cover problem
    U, V = get_set_cover_instance(Psi_dic, hat_E, tol=1e-10)

    avg_draws, all_drawn_subsets = get_random_set_cover_size(U, V, num_iterations)
    return avg_draws, all_drawn_subsets
