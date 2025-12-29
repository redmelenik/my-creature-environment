import numpy as np

# ==============================================================================
# 1. NEURAL NETWORK CLASS (The Brain/DNA)
# ==============================================================================

# Helper function to apply mutation safely

def _apply_mutation(matrix, rate, strength):
    """Applies mutation to a single NumPy matrix with explicit shape and type enforcement."""
    
    # 1. Force a fresh array copy with standard float dtype
    matrix_clean = np.array(matrix, dtype=np.float64, copy=True) 
    
    # 2. Force 2D shape. The .atleast_2d() is critical here.
    matrix_2d = np.atleast_2d(matrix_clean)
    
    # 3. CRITICAL: Handle the case where W1 is corrupted to (11, 1) or (11,) 
    # and must be (11, 12). This is a defensive catch for W1 (the largest weight matrix).
    # If the matrix is 2D but has the wrong size, we must fix it before getting the shape.
    if matrix_2d.ndim == 2 and matrix_2d.shape == (11, 1):
        # This catches a specific corruption where W1 (11x12) is seen as (11,1)
        matrix_2d = matrix_clean.reshape(11, 12).copy()
    elif matrix_2d.ndim == 1:
        # If it collapses to 1D (like the phantom (11,) shape), force it to be 2D
        matrix_2d = matrix_clean.reshape(1, -1).copy()

    # 4. Get the validated shape
    mutation_shape = matrix_2d.shape
    
    # 5. Create mask and mutation values explicitly matching the shape
    mask = np.random.rand(*mutation_shape) < rate
    mutation_values = np.random.randn(*mutation_shape) * strength
    
    # 6. Apply mutation
    # This line should now succeed.
    matrix_2d[mask] += mutation_values
    
    return matrix_2d # Return the mutated, clean array

class NeuralNetwork:
    """Represents the creature's Brain and DNA (weights/biases)."""

    INPUT_SIZE = 11
    HIDDEN_SIZE = 12
    OUTPUT_SIZE = 4 # Forward, Back, Left, Right

    def __init__(self, weights=None):
        if weights is None:
            self.W1 = np.random.randn(self.INPUT_SIZE, self.HIDDEN_SIZE) * 0.1
            self.b1 = np.zeros(self.HIDDEN_SIZE).reshape(1, -1) # Guarantee 2D
            self.W2 = np.random.randn(self.HIDDEN_SIZE, self.OUTPUT_SIZE) * 0.1
            self.b2 = np.zeros(self.OUTPUT_SIZE).reshape(1, -1) # Guarantee 2D
        else:
            # Load and guarantee 2D shape for all components
            self.W1 = np.atleast_2d(weights[0])
            self.b1 = np.atleast_2d(weights[1])
            self.W2 = np.atleast_2d(weights[2])
            self.b2 = np.atleast_2d(weights[3])

            # Ensure biases are (1, N) for sure
            if self.b1.shape[0] != 1: self.b1 = self.b1.reshape(1, -1)
            if self.b2.shape[0] != 1: self.b2 = self.b2.reshape(1, -1)

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
        Creates a slightly mutated copy of the parent DNA and applies mutation
        to four separate, independently named array copies to prevent memory bleed.
        """
        
        # 1. Unpack the DNA tuple into four distinct, fresh variables
        W1_copy = dna[0].copy()
        b1_copy = dna[1].copy()
        W2_copy = dna[2].copy()
        b2_copy = dna[3].copy()
        
        # 2. Apply mutation to each component explicitly
        W1_mutated = _apply_mutation(W1_copy, mutation_rate, mutation_strength) 
        b1_mutated = _apply_mutation(b1_copy, mutation_rate, mutation_strength) 
        W2_mutated = _apply_mutation(W2_copy, mutation_rate, mutation_strength) 
        b2_mutated = _apply_mutation(b2_copy, mutation_rate, mutation_strength) 
        
        # 3. Return the new tuple
        return (W1_mutated, b1_mutated, W2_mutated, b2_mutated)

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