import numpy as np
import itertools, time, random, math

from helper import compute_forward_distance, relative_angle

# generate polygon tower example : input : S, A, output : Phi, pi
def generate_polygon_tower(begin_index, end_index):

    # S is actual set of states and each state has been indexed from 0 to |S|-1 and the map is present in S_id_dic
    # All further indexing of state is done using S_id where s_id is the index of s in S.

    S = [idx for idx in range(begin_index, end_index+1, 1)]
    A = [idx for idx in range(0, max(S) + 1)]
    S_size, A_size = len(S), len(A)

    S_id_dic, Phi_dic, pi = {}, {}, {}

    for s_id, s in enumerate(S):
        S_id_dic[s_id] = s
        pi[s_id] = A_size-1             # the policy selects last action in each state
        for a in range(A_size):
            if(a == A_size-1):
                Phi_dic[(s_id, a)] = np.array([0,0,s])
            else:
                Phi_dic[(s_id, a)] = s*np.array([-np.cos(2*np.pi*a/s), -np.sin(2*np.pi*a/s), 0])

    return S, A, S_id_dic, Phi_dic, pi


def generate_pick_the_diamond(diamond_edges=[0,3,4,5,6], num_slots=6):

    def phi(s, a):
        return [a, s[a-1]]  # Subtract 1 from a because Python uses 0-based indexing

    def optimal_action(s):
        # Find the maximum value in the state
        max_value = max(s)
        
        # Find all indices with the max value and choose the highest index.
        # Add 1 because the indices are 1-based.
        return max([i+1 for i, val in enumerate(s) if val == max_value])
    
    S = list(itertools.product(diamond_edges, repeat=num_slots)) # a list of tuples each tuple is a state with num_slots elements
    empty_state = tuple([0] * num_slots)
    S.remove(empty_state)

    A = [idx for idx in range(1, num_slots+1)]

    S_id_dic, Phi_dic, pi = {}, {}, {}

    # Produce all possible phi's
    for s_id, s in enumerate(S):
        S_id_dic[s_id] = s
        pi[s_id] = optimal_action(s)
        for a in A:
            Phi_dic[(s_id,a)] = phi(s, a)

    return S, A, S_id_dic, Phi_dic, pi

def generate_grid_world_navigation(n, repeat, feature_type):
    
    def generate_board(n):
        board = [[' ' for _ in range(n)] for _ in range(n)]

        # Generate random positions for the agent and goal cells
        positions = random.sample([(i, j) for i in range(n) for j in range(n)], 2)
        agent_pos, goal_pos = positions

        # Generate random orientation for the agent
        orientations = ['U', 'D', 'L', 'R']
        agent_orientation = random.choice(orientations)

        # Place the agent and goal cells on the board
        board[agent_pos[0]][agent_pos[1]] = agent_orientation
        board[goal_pos[0]][goal_pos[1]] = 'G'

        return board, agent_pos, goal_pos, agent_orientation

    def generate_action_space(n=6, repeat=False):
        A = ['TA','TC', 'MV']

        if(repeat and n>3):
            A += ['MV'+str(i) for i in range(3,n)]
        
        return A
    
    def generate_all_board_states(n):
        board_states = []
        positions = [(i, j) for i in range(n) for j in range(n)]
        orientations = ['U', 'D', 'L', 'R']

        for agent_pos, goal_pos in itertools.permutations(positions, 2):
            for agent_orientation in orientations:
                board = [[' ' for _ in range(n)] for _ in range(n)]
                board[agent_pos[0]][agent_pos[1]] = agent_orientation
                board[goal_pos[0]][goal_pos[1]] = 'G'
                board_states.append((board, agent_pos, goal_pos, agent_orientation))

        return board_states
    
    def update_board(board, agent_pos, action):
        n = len(board)

        agent_orientation = board[agent_pos[0]][agent_pos[1]]

        # Update the agent's orientation based on the action
        orientations = ['L', 'D', 'R', 'U']

        if action == 'TC':
            agent_orientation = orientations[(orientations.index(agent_orientation) - 1) % 4]
        elif action == 'TA':
            agent_orientation = orientations[(orientations.index(agent_orientation) + 1) % 4]

        # Update the agent's position based on the move action
        elif 'MV' in action:

            repeat_count = 1
            if(action != 'MV'):
                repeat_count = int(action[2:])

            row, col = agent_pos
            board[row][col] = ' '

            if agent_orientation == 'U':
                row = max(0, row - repeat_count)
            elif agent_orientation == 'D':
                row = min(n - 1, row + repeat_count)
            elif agent_orientation == 'L':
                col = max(0, col - repeat_count)
            elif agent_orientation == 'R':
                board[row][col] = ' '
                col = min(n - 1, col + repeat_count)
            agent_pos = (row, col)

        else:
            raise ValueError('Invalid Action')

        # Update the board with the new agent position and orientation
        board[agent_pos[0]][agent_pos[1]] = agent_orientation

        return board, agent_pos

    def optimal_action(board, agent_pos, goal_pos, repeat):
        n = len(board)

        # Calculate the line of sight vector based on the agent's orientation
        agent_orientation = board[agent_pos[0]][agent_pos[1]]
        if agent_orientation == 'U':
            line_of_sight = (-1-agent_pos[0], 0)
        elif agent_orientation == 'D':
            line_of_sight = (n-agent_pos[0], 0)
        elif agent_orientation == 'L':
            line_of_sight = (0, -1-agent_pos[1])
        elif agent_orientation == 'R':
            line_of_sight = (0, n-agent_pos[1])

        # Calculate the direction vector from the agent to the goal
        goal_direction = (goal_pos[0] - agent_pos[0], goal_pos[1] - agent_pos[1])

        # Calculate the relative angle of goal direction from the line of sight in range [-pi, pi]
        angle = relative_angle(goal_direction, line_of_sight)

        # Determine the optimal action based on the angle
        if agent_pos == goal_pos:   # Check if agent and goal positions are the same
            return "TC"
        elif -math.pi / 2 < angle < math.pi / 2:
            if(repeat):
                forward_distance = compute_forward_distance(agent_pos, goal_pos, agent_orientation)
                if(forward_distance<=2):
                    return "MV"
                else:
                    return "MV"+str(forward_distance)
            else:
                return "MV"
        elif -math.pi < angle <= -math.pi / 2:
            return "TA"
        elif math.pi/2 <= angle <= math.pi or angle==-math.pi:
            return "TC"
        else:
            raise ValueError('Angle out of range [-pi,pi]')
    
    def local_feature_function(board, agent_pos, goal_pos, action):
        n = len(board)

        # Calculate the line of sight vector based on the agent's orientation
        agent_orientation = board[agent_pos[0]][agent_pos[1]]
        if agent_orientation == 'U':
            line_of_sight = (-1-agent_pos[0], 0)
        elif agent_orientation == 'D':
            line_of_sight = (n-agent_pos[0], 0)
        elif agent_orientation == 'L':
            line_of_sight = (0, -1-agent_pos[1])
        elif agent_orientation == 'R':
            line_of_sight = (0, n-agent_pos[1])

        # Calculate the direction vector from the agent to the goal
        goal_direction = (goal_pos[0] - agent_pos[0], goal_pos[1] - agent_pos[1])

        # Calculate the relative angle of goal direction from the line of sight in range [-pi,pi]
        angle = relative_angle(goal_direction, line_of_sight)

        # Define the feature functions
        features = [0] * 12

        if agent_pos == goal_pos:   # Check if agent and goal positions are the same
            features[0] = 1 if action == 'TA' else 0
            features[1] = 1 if action == 'TC' else 0
            features[2] = 1 if action == 'MV' else 0
        elif -math.pi < angle <= -math.pi / 2:
            features[3] = 1 if action == 'TA' else 0
            features[4] = 1 if action == 'TC' else 0
            features[5] = 1 if action == 'MV' else 0
        elif -math.pi / 2 < angle < math.pi / 2:
            features[6] = 1 if action == 'TA' else 0
            features[7] = 1 if action == 'TC' else 0
            features[8] = 1 if action == 'MV' else 0
        elif math.pi/2 <= angle <= math.pi or angle == -math.pi:
            features[9] = 1 if action == 'TA' else 0
            features[10] = 1 if action == 'TC' else 0
            features[11] = 1 if action == 'MV' else 0
        else:
            raise ValueError('Angle out of range [-pi,pi]')

        return features

    def local_feature_function_with_repeat(board, agent_pos, goal_pos, action):
        n = len(board)
        A = ['TA','TC'] + ['MV'+str(i) for i in range(1,n)]

        # Calculate the line of sight vector based on the agent's orientation
        agent_orientation = board[agent_pos[0]][agent_pos[1]]
        if agent_orientation == 'U':
            line_of_sight = (-1-agent_pos[0], 0)
        elif agent_orientation == 'D':
            line_of_sight = (n-agent_pos[0], 0)
        elif agent_orientation == 'L':
            line_of_sight = (0, -1-agent_pos[1])
        elif agent_orientation == 'R':
            line_of_sight = (0, n-agent_pos[1])

        # Calculate the direction vector from the agent to the goal
        goal_direction = (goal_pos[0] - agent_pos[0], goal_pos[1] - agent_pos[1])

        # Calculate the relative angle of goal direction from the line of sight in range [-pi,pi]
        angle = relative_angle(goal_direction, line_of_sight)
        forward_distance = compute_forward_distance(agent_pos, goal_pos, agent_orientation)

        # Define the feature functions
        features = [0] * (n**2 + 3*n + 2)

        if agent_pos == goal_pos:   # Check if agent and goal positions are the same
            current_index = 0
            for i, a in enumerate(A):
                features[current_index+i] = 1 if action == a else 0
        elif -math.pi < angle <= -math.pi / 2:
            current_index = len(A)
            for i, a in enumerate(A):
                features[current_index+i] = 1 if action == a else 0
        elif math.pi/2 <= angle <= math.pi or angle == -math.pi:
            current_index = 2*len(A)
            for i, a in enumerate(A):
                features[current_index+i] = 1 if action == a else 0
        elif -math.pi / 2 < angle < math.pi / 2:
            current_index = 3*len(A)
            for d in range(1, n):
                # if forward distance == d
                for i, a in enumerate(A):
                    features[current_index + (d-1)*len(A) + i] = 1 if forward_distance == d and action == a else 0
        else:
            raise ValueError('Angle out of range [-pi,pi]')

        return features

    def global_feature_function(board, agent_pos, goal_pos, action):
        start_row, start_col, orientation = agent_pos[0], agent_pos[1], board[agent_pos[0]][agent_pos[1]]
        goal_row, goal_col = goal_pos[0], goal_pos[1]

        feature_vector = [0] * 108

        # Type 1: Compare x-coordinate of agent and goal state
        for i in range(3):
            if (i == 0 and start_col == goal_col) or \
            (i == 1 and start_col > goal_col) or \
            (i == 2 and start_col < goal_col):
                for j in range(3):
                    for k in range(4):
                        for l in range(3):
                            feature_index = i * 36 + j * 12 + k * 3 + l
                            feature_vector[feature_index] = 1

        # Type 2: Compare y-coordinate of agent and goal state
        for j in range(3):
            if (j == 0 and start_row == goal_row) or \
            (j == 1 and start_row > goal_row) or \
            (j == 2 and start_row < goal_row):
                for i in range(3):
                    for k in range(4):
                        for l in range(3):
                            feature_index = i * 36 + j * 12 + k * 3 + l
                            feature_vector[feature_index] = 1

        # Type 3: Check the current orientation of the agent
        for k in range(4):
            if (k == 0 and orientation == 'L') or \
            (k == 1 and orientation == 'R') or \
            (k == 2 and orientation == 'U') or \
            (k == 3 and orientation == 'D'):
                for i in range(3):
                    for j in range(3):
                        for l in range(3):
                            feature_index = i * 36 + j * 12 + k * 3 + l
                            feature_vector[feature_index] = 1

        # Type 4: Check the current action
        for l in range(3):
            if (l == 0 and action == 'TA') or \
            (l == 1 and action == 'TC') or \
            (l == 2 and action == 'MV'):
                for i in range(3):
                    for j in range(3):
                        for k in range(4):
                            feature_index = i * 36 + j * 12 + k * 3 + l
                            feature_vector[feature_index] = 1

        return feature_vector

    def global_feature_function_with_repeat(board, agent_pos, goal_pos, agent_action):

        n, agent_orientation = len(board), board[agent_pos[0]][agent_pos[1]]

        # Feature vector of size 3(x_compare) * 3(y_compare) * 4(agent_orient) * (n-1)(forward_distance) * (n+1)(actions)
        if(n<=3):
            features_size = 3 * 3 * 4 * (n-1) * 3       # only three actions [TA, TC, MV]
        else:
            features_size = 3 * 3 * 4 * (n-1) * n       # n actions [TA, TC, MV, MV3, ... , MVn-1]
        features = [0] * features_size

        # Extract agent's and goal's x and y coordinates
        ax, ay = agent_pos
        gx, gy = goal_pos

        forward_distance = compute_forward_distance(agent_pos, goal_pos, agent_orientation)

        # Map orientation and action to index
        orientation_index = {'U': 0, 'D': 1, 'L': 2, 'R': 3}
        action_index = {'TA': 0, 'TC': 1, 'MV': 2}

        # Including all possible move actions
        if(n>=4):
            for i in range(3, n):
                action_index[f'MV{i}'] = i

        # Determine the indexes for x and y comparisons
        x_comp = (ax > gx) * 0 + (ax < gx) * 1 + (ax == gx) * 2
        y_comp = (ay > gy) * 0 + (ay < gy) * 1 + (ay == gy) * 2

        # Get the orientation and action indexes
        orient = orientation_index[board[ax][ay]]
        action = action_index[agent_action]

        # Calculate the feature index, forward_distance should be offset by -1 to fit the 0-indexed array
        # Formula adjustment: forward_distance - 1
        index = ((x_comp * 3 + y_comp) * 4 * (n - 1) * n + orient * (n - 1) * n + (forward_distance - 1) * n + action)
        features[index] = 1

        return features

    S = generate_all_board_states(n)
    A = generate_action_space(n, repeat)

    S_id_dic, Phi_dic, pi = {}, {}, {}
    for s_id, s in enumerate(S):

        S_id_dic[s_id] = s
        # compute optimal action at this state
        pi[s_id] = optimal_action(s[0], s[1], s[2], repeat)
        # compute feature function of optimal action

        if(feature_type=='global'):
            if(repeat):
                feature_function = global_feature_function_with_repeat
            else:
                feature_function = global_feature_function
        elif(feature_type=='local'):
            if(repeat):
                feature_function = local_feature_function_with_repeat
            else:
                feature_function = local_feature_function
        else:
            raise ValueError("Wrong feature function type!")

        Phi_dic[(s_id, pi[s_id])] = feature_function(s[0], s[1], s[2], pi[s_id])


        for action in A:
            # compute feature function of non-optimal action and corresponding feature difference vectors
            if(action!=pi[s_id]):
                Phi_dic[(s_id,action)] = feature_function(s[0], s[1], s[2], action)

    return S, A, S_id_dic, Phi_dic, pi

# create feature difference tensor from feature tensor and deterministic policy
def get_feature_differences(S, A, Phi_dic, pi):
    
    # compute feature difference set Psi
    Psi_dic = {}

    for s_id, s in enumerate(S):
        phi_opt = Phi_dic[(s_id,pi[s_id])]
        for a in A:
            if(a!=pi[s_id]):
                Psi_dic[(s_id, a)] = [x - y for x, y in zip(phi_opt, Phi_dic[(s_id,a)])]

    return Psi_dic