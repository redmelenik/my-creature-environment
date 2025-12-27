import numpy as np

class NeuralNetwork:
    """Represents the creature's Brain and DNA (weights/biases)."""

    INPUT_SIZE = 11
    HIDDEN_SIZE = 12
    OUTPUT_SIZE = 4 # Forward, Back, Left, Right

    def __init__(self, weights=None):
        if weights is None:
            # Initialize random weights (W1: Input -> Hidden, W2: Hidden -> Output)
            # Biases are included as part of the weights for simplicity (often cleaner)
            self.W1 = np.random.randn(self.INPUT_SIZE, self.HIDDEN_SIZE) * 0.1
            self.b1 = np.zeros((1, self.HIDDEN_SIZE))
            self.W2 = np.random.randn(self.HIDDEN_SIZE, self.OUTPUT_SIZE) * 0.1
            self.b2 = np.zeros((1, self.OUTPUT_SIZE))
        else:
            self.W1, self.b1, self.W2, self.b2 = weights

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        """Processes the environment state to determine the action."""
        # Convert input list/array to a (1, INPUT_SIZE) numpy array
        X = np.array(inputs).reshape(1, -1)

        # Layer 1: Input -> Hidden
        L1 = np.dot(X, self.W1) + self.b1
        H = self.sigmoid(L1)

        # Layer 2: Hidden -> Output
        L2 = np.dot(H, self.W2) + self.b2
        Outputs = L2 # Using linear output for simple action selection

        # The action is the index of the highest output value
        action_index = np.argmax(Outputs)
        return action_index # 0: Fwd, 1: Back, 2: Left, 3: Right

    def get_dna(self):
        """Returns the creature's DNA (all weights and biases)."""
        return (self.W1, self.b1, self.W2, self.b2)

    @classmethod
    def mutate(cls, dna, mutation_rate=0.1, mutation_strength=0.1):
        """Creates a slightly mutated copy of the parent DNA."""
        W1, b1, W2, b2 = [d.copy() for d in dna] # Deep copy

        for matrix in [W1, b1, W2, b2]:
            # Apply mutation to a small percentage of elements
            mask = np.random.rand(*matrix.shape) < mutation_rate
            mutation_values = np.random.randn(*matrix.shape) * mutation_strength
            matrix[mask] += mutation_values

        return (W1, b1, W2, b2)
        
        class Creature:
    """Represents an individual creature."""

    # Target Golden Ratio
    TARGET_BRAIN, TARGET_BODY, TARGET_LEGS = 25, 20, 16

    # Death Limits
    MAX_BRAIN, MAX_BODY, MAX_LEGS = 30, 25, 20

    # Token constraints for evolution sequence
    BRAIN_REQ_FOR_BODY = 5
    BODY_REQ_FOR_LEGS = 5
    LEG_RATIO_PER_BODY = 4

    def __init__(self, world_size, dna=None):
        self.world_size = world_size
        self.gender = np.random.choice(['M', 'F'])
        self.position = np.random.rand(2) * world_size # [x, y] in [0, world_size)

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
        self.touch_counter = {} # {Creature_ID: count, ...}
        self.tribe_id = None
        self.speech_tokens = 0

    def get_speed(self, direction):
        """Calculates speed based on collected leg sub-tokens."""
        # Max speed is 1.0, scaled by the specific directional token count (max 4)
        dir_map = {0: 'fwd', 1: 'back', 2: 'left', 3: 'right'}
        key = dir_map.get(direction, 'fwd')
        return 0.1 * (self.leg_sub_tokens[key] / 4.0)

    def move(self, action_index):
        """Applies movement based on the Neural Network output."""
        if not self.alive: return

        speed = 0.1 # Base speed

        # 0: Forward, 1: Backward, 2: Left, 3: Right
        if action_index == 0: # Forward (e.g., +Y)
            self.position[1] += self.get_speed(0)
        elif action_index == 1: # Backward (e.g., -Y)
            self.position[1] -= self.get_speed(1)
        elif action_index == 2: # Left (e.g., -X)
            self.position[0] -= self.get_speed(2)
        elif action_index == 3: # Right (e.g., +X)
            self.position[0] += self.get_speed(3)

        # Enforce boundary conditions (stay within the world square)
        self.position = np.clip(self.position, 0, self.world_size - 1e-6)

    def collect_token(self, token_type):
        """Applies the complex token collection logic."""
        if not self.alive: return

        # 1. Check constraints for collection
        can_collect = False
        if token_type == 'brain' and self.brain_tokens < self.MAX_BRAIN:
            can_collect = True
            self.brain_tokens += 1
        elif token_type == 'body':
            if self.brain_tokens >= self.BRAIN_REQ_FOR_BODY and self.body_tokens < self.MAX_BODY:
                can_collect = True
                self.body_tokens += 1
        elif token_type == 'legs':
            if self.body_tokens >= self.BODY_REQ_FOR_LEGS and self.leg_tokens < self.MAX_LEGS:
                # Check Leg-to-Body Ratio: must have 4 legs for every 5 body
                # The total number of legs collected *must not* exceed (body_tokens // 5) * 4
                max_legs_allowed = (self.body_tokens // self.BODY_REQ_FOR_LEGS) * self.LEG_RATIO_PER_BODY
                if self.leg_tokens < max_legs_allowed:
                    can_collect = True
                    self.leg_tokens += 1
                    # Assign to a random leg sub-token for directional movement
                    sub_token_key = np.random.choice(list(self.leg_sub_tokens.keys()))
                    if self.leg_sub_tokens[sub_token_key] < 4: # Max 4 per direction
                         self.leg_sub_tokens[sub_token_key] += 1
        
        if can_collect:
            # 2. Check for immediate death after collection
            if self.brain_tokens > self.MAX_BRAIN or \
               self.body_tokens > self.MAX_BODY or \
               self.leg_tokens > self.MAX_LEGS:
                self.alive = False
                # print(f"Creature died from excess {token_type} tokens.")
            return True
        return False

    def calculate_fitness(self):
        """Calculates strength based on proximity to the Golden Ratio."""
        if not self.alive:
            self.fitness = 0.0
            return

        # Sum of absolute differences from the target ratio
        diff = abs(self.brain_tokens - self.TARGET_BRAIN) + \
               abs(self.body_tokens - self.TARGET_BODY) + \
               abs(self.leg_tokens - self.TARGET_LEGS)

        # Fitness is the reciprocal of the difference (+1 to avoid Div/0 and reward max)
        self.fitness = 1.0 / (diff + 1.0)
        
        class Environment:
    def __init__(self, world_size=100, num_creatures=100, token_count=50):
        self.WORLD_SIZE = world_size
        self.NUM_CREATURES = num_creatures
        self.TOKEN_COUNT = token_count
        self.generation = 1
        self.fittest_dna = None # DNA of the strongest creature
        self.creatures = []
        self.tokens = self._generate_tokens()

    def _generate_tokens(self):
        """Generates random positions and types for food tokens."""
        tokens = []
        token_types = ['brain'] * 20 + ['body'] * 15 + ['legs'] * 15 # Biased distribution
        np.random.shuffle(token_types)
        
        for t_type in token_types[:self.TOKEN_COUNT]:
            tokens.append({
                'type': t_type,
                'position': np.random.rand(2) * self.WORLD_SIZE,
                'collected': False
            })
        return tokens

    def _initialize_creatures(self):
        """Spawns or inherits creatures for the new generation."""
        self.creatures = []
        
        # 1. The Fittest Creature from the previous generation survives/is cloned
        if self.fittest_dna:
            # Clone the fittest from the previous era
            self.creatures.append(Creature(self.WORLD_SIZE, self.fittest_dna))
            # The rest are mutated copies
            for _ in range(self.NUM_CREATURES - 1):
                mutated_dna = NeuralNetwork.mutate(self.fittest_dna)
                self.creatures.append(Creature(self.WORLD_SIZE, mutated_dna))
        else:
            # Generation 1: all random DNA
            for _ in range(self.NUM_CREATURES):
                self.creatures.append(Creature(self.WORLD_SIZE))

    def _find_closest_token(self, creature):
        """Helper to find the nearest uncollected food token."""
        min_dist = np.inf
        closest_token_pos = None
        closest_token_type = None

        for token in self.tokens:
            if not token['collected']:
                dist = np.linalg.norm(creature.position - token['position'])
                if dist < min_dist:
                    min_dist = dist
                    closest_token_pos = token['position']
                    closest_token_type = token['type']
                    
        # Return normalized coordinates and token type
        if closest_token_pos is not None:
            # Normalized position
            norm_pos = closest_token_pos / self.WORLD_SIZE
            return norm_pos[0], norm_pos[1], closest_token_type, min_dist
        
        # Default return if no tokens left
        return 0.5, 0.5, 'none', np.inf

    def run_generation(self, time_steps=100):
        """Runs the simulation for one full generation."""
        self._initialize_creatures()
        self.tokens = self._generate_tokens() # New tokens for the new era
        
        print(f"--- Starting Generation {self.generation} with {len(self.creatures)} Creatures ---")

        for step in range(time_steps):
            alive_count = sum(c.alive for c in self.creatures)
            if alive_count == 0:
                print(f"All creatures died at step {step}. Ending generation early.")
                break

            for i, creature in enumerate(self.creatures):
                if not creature.alive: continue

                # --- 1. SENSORY INPUT ---
                # Find the closest token (Brain priority handled implicitly by NN later)
                tx, ty, t_type, dist_to_token = self._find_closest_token(creature)

                # Prepare input for the Neural Network
                inputs = [
                    tx, ty,                                                   # Closest Token position
                    creature.position[0] / self.WORLD_SIZE,                   # Creature X (normalized)
                    creature.position[1] / self.WORLD_SIZE,                   # Creature Y (normalized)
                    creature.brain_tokens / Creature.MAX_BRAIN,               # Normalized token counts
                    creature.body_tokens / Creature.MAX_BODY,
                    creature.leg_tokens / Creature.MAX_LEGS,
                    Creature.TARGET_BRAIN, Creature.TARGET_BODY, Creature.TARGET_LEGS, # Target Ratios (fixed)
                    len(creature.touch_counter) / self.NUM_CREATURES          # Touch Count
                ]

                # --- 2. BRAIN PROCESS & ACTION ---
                action = creature.nn.forward(inputs)
                creature.move(action)

                # --- 3. ENVIRONMENT INTERACTION (Token Collection) ---
                for token in self.tokens:
                    if not token['collected']:
                        # Simple collision check (distance threshold)
                        collision_dist = np.linalg.norm(creature.position - token['position'])
                        if collision_dist < 2.0: # Collision threshold
                            if creature.collect_token(token['type']):
                                token['collected'] = True
                                # print(f"Creature {i} collected a {token['type']} token.")

                # --- 4. TRIBE/SPEECH INTERACTION (Simplified Contact) ---
                for other in self.creatures:
                    if creature != other and other.alive:
                         # Simple contact check
                        contact_dist = np.linalg.norm(creature.position - other.position)
                        if contact_dist < 1.0:
                            # Update touch counter
                            creature.touch_counter[id(other)] = creature.touch_counter.get(id(other), 0) + 1

                            # Check for tribe creation
                            if creature.touch_counter[id(other)] >= 5 and creature.tribe_id is None:
                                # Simple tribe creation: assign a unique ID or inherit
                                if other.tribe_id is not None:
                                    creature.tribe_id = other.tribe_id
                                else:
                                    # New tribe ID (using creature's ID as a temporary ID)
                                    creature.tribe_id = id(creature)
                                    other.tribe_id = id(creature)
                                # Speech Token collected upon tribe formation
                                creature.speech_tokens += 1
                                other.speech_tokens += 1

        # --- 5. FITNESS EVALUATION & SELECTION ---
        for creature in self.creatures:
            creature.calculate_fitness()

        # Find the fittest creature
        fittest = max(self.creatures, key=lambda c: c.fitness)
        self.fittest_dna = fittest.nn.get_dna()

        print(f"\n--- Generation {self.generation} Results ---")
        print(f"Fittest Score: {fittest.fitness:.4f}")
        print(f"Tokens: B:{fittest.brain_tokens}, D:{fittest.body_tokens}, L:{fittest.leg_tokens}")
        print(f"Leg Directions: {fittest.leg_sub_tokens}")
        print(f"Speech Tokens: {fittest.speech_tokens}")
        
        self.generation += 1
        return fittest.fitness

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # 
    # This image would help visualize the abstract "world" the code creates.
    
    # Initialize the environment
    env = Environment(world_size=100, num_creatures=100)
    
    # Run multiple generations
    num_generations = 10
    
    for _ in range(num_generations):
        best_fitness = env.run_generation(time_steps=200)

        # Check for convergence (perfect fitness)
        if best_fitness >= 1.0:
            print("\n!!! Goal Achieved: A creature reached perfect fitness (Golden Ratio) !!!")
            break
            
            