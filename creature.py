import numpy as np

# ==============================================================================
# 1. NEURAL NETWORK CLASS (The Brain/DNA)
# ==============================================================================

class NeuralNetwork:
    """Represents the creature's Brain and DNA (weights/biases)."""

    INPUT_SIZE = 11
    HIDDEN_SIZE = 12
    OUTPUT_SIZE = 4 # Forward, Back, Left, Right

    def __init__(self, weights=None):
        if weights is None:
            # W1: (11, 12)
            self.W1 = np.random.randn(self.INPUT_SIZE, self.HIDDEN_SIZE) * 0.1
            # b1: MUST BE (1, 12)
            self.b1 = np.zeros((1, self.HIDDEN_SIZE)) 
            # W2: (12, 4)
            self.W2 = np.random.randn(self.HIDDEN_SIZE, self.OUTPUT_SIZE) * 0.1
            # b2: MUST BE (1, 4)
            self.b2 = np.zeros((1, self.OUTPUT_SIZE)) 
        else:
            self.W1, self.b1, self.W2, self.b2 = weights

    def sigmoid(self, x):
        """Standard sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        """Processes the environment state to determine the action."""
        X = np.array(inputs).reshape(1, -1)

        # Layer 1
        L1 = np.dot(X, self.W1) + self.b1
        H = self.sigmoid(L1)

        # Layer 2
        L2 = np.dot(H, self.W2) + self.b2
        
        action_index = np.argmax(L2) 
        return action_index

    def get_dna(self):
        """Returns the creature's DNA (all weights and biases)."""
        return (self.W1, self.b1, self.W2, self.b2)

    @classmethod
    def mutate(cls, dna, mutation_rate=0.1, mutation_strength=0.1):
        """
        Creates a slightly mutated copy of the parent DNA.
        The shape corruption is fixed by using indexing to modify the list 
        of copies, ensuring NumPy maintains the matrix structure.
        """
        
        # 1. Create a LIST of deep copies
        dna_list = [d.copy() for d in dna] 

        # 2. Iterate through the list using index to ensure correct reassignment
        for i, matrix in enumerate(dna_list):
            
            # Ensure the matrix is 2D
            matrix_2d = np.atleast_2d(matrix)

            # Generate mask and mutation values based on the 2D shape
            mask = np.random.rand(*matrix_2d.shape) < mutation_rate
            mutation_values = np.random.randn(*matrix_2d.shape) * mutation_strength
            
            # Apply mutation
            matrix_2d[mask] += mutation_values
            
            # CRITICAL: Re-assign the potentially modified matrix back into the list
            # Since matrix_2d is a view, this might not be strictly necessary, 
            # but it is a strong defense against corruption.
            dna_list[i] = matrix_2d

        # Return the list as a tuple
        return tuple(dna_list)

# ==============================================================================
# 2. CREATURE CLASS
# ==============================================================================

class Creature:
    """Represents an individual creature with state, movement, and resource logic."""

    # --- CONSTANTS ---
    
    # Target Golden Ratio for perfect evolution
    TARGET_BRAIN, TARGET_BODY, TARGET_LEGS = 25, 20, 16

    # Death Limits
    MAX_BRAIN, MAX_BODY, MAX_LEGS = 30, 25, 20

    # Token constraints for evolution sequence
    BRAIN_REQ_FOR_BODY = 5
    BODY_REQ_FOR_LEGS = 5
    LEG_RATIO_PER_BODY = 4 # Max 4 legs for every 5 body tokens

    def __init__(self, world_size, dna=None):
        self.world_size = world_size
        self.gender = np.random.choice(['M', 'F'])
        self.position = np.random.rand(2) * world_size # [x, y] coordinates

        # Characteristics (Token Counts)
        self.brain_tokens = 0
        self.body_tokens = 0
        self.leg_tokens = 0
        
        # Leg Sub-Tokens for directional movement
        self.leg_sub_tokens = {'fwd': 0, 'back': 0, 'left': 0, 'right': 0}

        self.nn = NeuralNetwork(dna)
        self.fitness = 0.0
        self.alive = True
        
        # Tribe/Communication
        # Using a tuple (creature_id, creature_id) as key for mutual touch
        self.touch_counter = {} 
        self.tribe_id = None
        self.speech_tokens = 0

    def get_speed(self, action_index):
        """Calculates movement speed based on collected leg sub-tokens."""
        dir_map = {0: 'fwd', 1: 'back', 2: 'left', 3: 'right'}
        key = dir_map.get(action_index)
        
        # Base movement scalar (e.g., 0.1) scaled by the collected sub-tokens (max 4)
        base_speed = 0.5
        if key in self.leg_sub_tokens:
            # Scale speed based on token completion (0 to 1.0)
            return base_speed * (self.leg_sub_tokens[key] / 4.0)
        return 0.0 # Should not happen

    def move(self, action_index):
        """Applies movement based on the Neural Network output and leg tokens."""
        if not self.alive: return

        speed = self.get_speed(action_index)

        # 0: Forward (+Y), 1: Backward (-Y), 2: Left (-X), 3: Right (+X)
        if action_index == 0: 
            self.position[1] += speed
        elif action_index == 1: 
            self.position[1] -= speed
        elif action_index == 2: 
            self.position[0] -= speed
        elif action_index == 3: 
            self.position[0] += speed

        # Enforce boundary conditions
        self.position = np.clip(self.position, 0, self.world_size - 1e-6)

    def collect_token(self, token_type):
        """Applies the complex token collection logic and death check."""
        if not self.alive: return

        can_collect = False
        
        if token_type == 'brain':
            if self.brain_tokens < self.MAX_BRAIN:
                can_collect = True
                self.brain_tokens += 1
        
        elif token_type == 'body':
            if self.brain_tokens >= self.BRAIN_REQ_FOR_BODY and self.body_tokens < self.MAX_BODY:
                can_collect = True
                self.body_tokens += 1
        
        elif token_type == 'legs':
            if self.body_tokens >= self.BODY_REQ_FOR_LEGS and self.leg_tokens < self.MAX_LEGS:
                # Check Leg-to-Body Ratio Constraint
                max_legs_allowed = (self.body_tokens // self.BODY_REQ_FOR_BODY) * self.LEG_RATIO_PER_BODY
                
                if self.leg_tokens < max_legs_allowed:
                    can_collect = True
                    self.leg_tokens += 1
                    
                    # Assign to the least-developed leg sub-token (prioritizing full movement)
                    min_key = min(self.leg_sub_tokens, key=self.leg_sub_tokens.get)
                    if self.leg_sub_tokens[min_key] < 4:
                         self.leg_sub_tokens[min_key] += 1
        
        if can_collect:
            # Check for death limits
            if self.brain_tokens > self.MAX_BRAIN or \
               self.body_tokens > self.MAX_BODY or \
               self.leg_tokens > self.MAX_LEGS:
                self.alive = False
            return True
            
        return False

    def calculate_fitness(self):
        """Calculates strength based on proximity to the Golden Ratio (25:20:16)."""
        if not self.alive:
            self.fitness = 0.0
            return

        # Sum of absolute differences from the target ratio
        diff = abs(self.brain_tokens - self.TARGET_BRAIN) + \
               abs(self.body_tokens - self.TARGET_BODY) + \
               abs(self.leg_tokens - self.TARGET_LEGS)

        # Fitness is the reciprocal of the difference (+1 to avoid Div/0 and reward max)
        self.fitness = 1.0 / (diff + 1.0)