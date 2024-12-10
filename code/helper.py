import numpy as np
import math

def l2_norm(x):
  return np.sqrt(np.dot(x,x))

def is_equal_vectors(v1, v2, tol=1e-10):
    # using absolute tolerance to compare two vectors |v1_i - v2_i| <= tol
    tol = float(tol)
    return np.all(np.abs(v1 - v2) < tol)

def print_board(board):
    n = len(board)
    print("   " + "  ".join(str(i) for i in range(n)))
    print("  +" + "-" * (3 * n - 1) + "+")
    for i in range(n):
        row = board[i]
        print(f"{i} |" + "|".join(cell.center(2) for cell in row) + "|")
        if i < n - 1:
            print("  +" + "-" * (3 * n - 1) + "+")
    print("  +" + "-" * (3 * n - 1) + "+")

def print_state(s, instance_type):
    if('polygon' in instance_type):
        print("State : ", s)
    elif(instance_type=='pick_the_diamond'):
        print("State : ", s)
    elif(instance_type=='grid_world_navigation'):
        print("Board Configuration : ")
        print_board(s[0])
        pass
    else:
        raise ValueError('Wrong instance type!')

def get_unique_vectors(X, tol=1e-10):
    """
    Get unique vectors among rows of a matrix X.

    Args:
        X (numpy.ndarray): Input matrix of shape (n, d).
        tol (float, optional): Absolute tolerance for comparing components. Default is 1e-10.

    Returns:
        numpy.ndarray: Array of unique vectors.
    """

    # Get the number of rows in the matrix
    n = X.shape[0]

    # Create a boolean mask to track unique vectors
    mask = np.ones(n, dtype=bool)

    # Iterate over each pair of vectors
    for i in range(n):
        if mask[i]:
            for j in range(i+1, n):
                if mask[j] and is_equal_vectors(X[i], X[j], tol):
                    mask[j] = False

    # Return the unique vectors
    return X[mask]


# create unique rows from a feature difference tensor
def get_unique_normalized_vectors(Psi, tol=1e-10):

    # input : 
    #   1. Psi - a S x A-1 x d dimensional array
    # output :
    #   1. unique_hat_X - a k x d dimensional array, where each row is a unqiue normalized feature vector of Psi
    
    if(len(Psi.shape)>=2):    # used when Psi is S x A x d dimensional array
        Psi = Psi.reshape(-1, Psi.shape[-1])
    
    hat_Psi = Psi/np.linalg.norm(Psi, axis=1, keepdims=True)
    unique_hat_Psi = get_unique_vectors(hat_Psi, tol=tol)
    return unique_hat_Psi


def process_text(X):
    if(len(X.shape)==1):
        text = "minimize z: {0:2.4f}*x1 + {1:2.4f}*x2 + {2:2.4f}*x3;".format(X[0], X[1], X[2])
    
    else:
        text = ""
        for i in range(X.shape[0]):
            text += 'subject to c{0:1d}: {1:2.4f}*x1 + {2:2.4f}*x2 + {3:2.4f}*x3 <= 1;'.format(i, X[i][0], X[i][1], X[i][2])+'\n'
    
    return text



# relative angle of v1 from v2
def relative_angle(v1, v2):
    # Calculate the angle using atan2 and normalize in [-pi,pi] range
    angle_rad = math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])
    angle_rad = (angle_rad + math.pi) % (2 * math.pi) - math.pi
    return angle_rad

# compute the forward distance between agent and goal
def compute_forward_distance(agent_pos, goal_pos, agent_orientation):
    if(agent_orientation=='U' or agent_orientation=='D'):
      return abs(agent_pos[0]-goal_pos[0])

    if(agent_orientation=='L' or agent_orientation=='R'):
      return abs(agent_pos[1]-goal_pos[1])
