import numpy as np
import uuid 
from creature import Creature, NeuralNetwork # Assuming your class definitions are in creature.py

class Environment:
    """Manages the simulation, including world boundaries, tokens, and generations."""
    
    def __init__(self, world_size=100, num_creatures=100, initial_token_count=50):
        self.WORLD_SIZE = world_size
        self.NUM_CREATURES = num_creatures
        self.INITIAL_TOKEN_COUNT = initial_token_count
        self.current_generation = 1
        self.fittest_dna = None # Stores the DNA of the best creature from the last era
        self.creatures = []
        self.tokens = []

def _generate_tokens(self):
    """Generates random positions and types for food tokens."""
    self.tokens = []
    # Bias the tokens towards Brain, then Body, then Legs
    token_types = ['brain'] * 25 + ['body'] * 15 + ['legs'] * 10
    
    # Only use the number of tokens defined by INITIAL_TOKEN_COUNT
    token_sample = np.random.choice(token_types, size=self.INITIAL_TOKEN_COUNT, replace=False)
    
    for t_type in token_sample:
        self.tokens.append({
            'type': t_type,
            'position': np.random.rand(2) * self.WORLD_SIZE,
            'collected': False
        })

        # 1. Spawn the "Seed" creature (cloned from the fittest)
        if self.fittest_dna:
            # Unpack the 6-element DNA, including original shapes
            W1_flat, b1, W2_flat, b2, W1_shape, W2_shape = self.fittest_dna

            # Rebuild W1 and W2 from the flattened data with guaranteed 2D shape
            W1 = W1_flat.reshape(W1_shape)
            W2 = W2_flat.reshape(W2_shape)

            # Define the clean 4-component DNA tuple for use
            clean_dna = (W1, b1, W2, b2)
            
            # Clone the fittest
            self.creatures.append(Creature(self.WORLD_SIZE, clean_dna))
            
            # The rest are mutated copies (Offspring)
            for _ in range(self.NUM_CREATURES - 1):
                mutated_dna = NeuralNetwork.mutate(clean_dna)
                self.creatures.append(Creature(self.WORLD_SIZE, mutated_dna))
        else:
            # Generation 1: all creatures have random DNA
            for _ in range(self.NUM_CREATURES):
                self.creatures.append(Creature(self.WORLD_SIZE))
        
        # Assign unique IDs for tribal tracking
        for creature in self.creatures:
             creature.id = uuid.uuid4()
             
    # In evolutionary_sim.py (inside the Environment class)

def _initialize_creatures(self):
    """Spawns 100 creatures, inheriting or mutating the fittest DNA."""
    self.creatures = []
    
    # 1. Spawn the "Seed" creature (cloned from the fittest)
    if self.fittest_dna:
        
        # CRITICAL FIX: Unpack 4 components (W1, b1, W2, b2) 
        # The 6-element storage was removed in a previous step.
        W1, b1, W2, b2 = self.fittest_dna

        # Define the clean 4-component DNA tuple for use, forcing copies for absolute safety
        clean_dna = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
        
        # Clone the fittest
        self.creatures.append(Creature(self.WORLD_SIZE, clean_dna))
        
        # The rest are mutated copies (Offspring)
        for _ in range(self.NUM_CREATURES - 1):
            mutated_dna = NeuralNetwork.mutate(clean_dna)
            self.creatures.append(Creature(self.WORLD_SIZE, mutated_dna))
    else:
        # Generation 1: all creatures have random DNA
        for _ in range(self.NUM_CREATURES):
            self.creatures.append(Creature(self.WORLD_SIZE))
    
    # Assign unique IDs for tribal tracking
    import uuid # Ensure uuid is imported if not already at the top of the file
    for creature in self.creatures:
         creature.id = uuid.uuid4()

    def _find_closest_token_pos(self, creature):
        """Finds the normalized position of the closest uncollected token."""
        min_dist = np.inf
        closest_token_pos = None

        for token in self.tokens:
            if not token['collected']:
                # Calculate Euclidean distance
                dist = np.linalg.norm(creature.position - token['position'])
                if dist < min_dist:
                    min_dist = dist
                    closest_token_pos = token['position']
                    
        # Return normalized coordinates (0 to 1)
        if closest_token_pos is not None:
            norm_pos = closest_token_pos / self.WORLD_SIZE
            return norm_pos[0], norm_pos[1]
        
        # Default return if no tokens left (center of the world)
        return 0.5, 0.5

    def _handle_interactions(self, creature):
        """Handles collision with tokens and other creatures."""
        
        # Token Interaction (Collection)
        for token in self.tokens:
            if not token['collected']:
                collision_dist = np.linalg.norm(creature.position - token['position'])
                if collision_dist < 1.0: # Simple collision threshold
                    if creature.collect_token(token['type']):
                        token['collected'] = True
                        
        # Creature Interaction (Tribe/Speech)
        for other in self.creatures:
            if creature != other and other.alive:
                contact_dist = np.linalg.norm(creature.position - other.position)
                if contact_dist < 1.0: # Simple contact threshold
                    
                    # Mutual touch counter update
                    touch_key = tuple(sorted((creature.id, other.id)))
                    creature.touch_counter[touch_key] = creature.touch_counter.get(touch_key, 0) + 1
                    
                    # Check for tribe creation/joining
                    if creature.touch_counter[touch_key] >= 5:
                        
                        # Assign/Propagate Tribe ID
                        if creature.tribe_id is None and other.tribe_id is None:
                            # New tribe
                            tribe_id = uuid.uuid4()
                            creature.tribe_id = tribe_id
                            other.tribe_id = tribe_id
                            creature.speech_tokens += 1
                            other.speech_tokens += 1
                        elif creature.tribe_id is None:
                            # Join existing tribe
                            creature.tribe_id = other.tribe_id
                        elif other.tribe_id is None:
                            # Join existing tribe
                            other.tribe_id = creature.tribe_id

    def run_generation(self, time_steps=200):
        """Runs the simulation for one full generation (era)."""
        self._initialize_creatures()
        self._generate_tokens() 
        
        print(f"--- Starting Generation {self.current_generation} ---")
        
        for step in range(time_steps):
            alive_count = sum(c.alive for c in self.creatures)
            if alive_count == 0:
                print(f"All creatures died at step {step}. Ending generation early.")
                break

            for creature in self.creatures:
                if not creature.alive: continue

                # 1. SENSORY INPUT
                tx, ty = self._find_closest_token_pos(creature)
                
                # Input normalization
                inputs = [
                    tx, ty,                                                   
                    creature.position[0] / self.WORLD_SIZE,                   
                    creature.position[1] / self.WORLD_SIZE,                   
                    creature.brain_tokens / Creature.MAX_BRAIN,               
                    creature.body_tokens / Creature.MAX_BODY,
                    creature.leg_tokens / Creature.MAX_LEGS,
                    Creature.TARGET_BRAIN, Creature.TARGET_BODY, Creature.TARGET_LEGS, 
                    len(creature.touch_counter) / self.NUM_CREATURES          
                ]

                # 2. BRAIN PROCESS & ACTION
                action = creature.nn.forward(inputs)
                creature.move(action)

                # 3. INTERACTION
                self._handle_interactions(creature)

        # 4. FITNESS EVALUATION & SELECTION
        for creature in self.creatures:
            creature.calculate_fitness()

        fittest = max(self.creatures, key=lambda c: c.fitness)
        # FINAL SIMPLE STORAGE: Store the standard 4-component DNA tuple
        self.fittest_dna = fittest.nn.get_dna() 
        # NEW: Store W1 and W2 as flattened arrays (data only)
        # W1_flat = fittest.nn.W1.flatten()
        # W2_flat = fittest.nn.W2.flatten()

        # Store the raw data, preserving the original shapes as metadata
        # self.fittest_dna = (W1_flat, fittest.nn.b1, W2_flat, fittest.nn.b2, 
        #                    fittest.nn.W1.shape, fittest.nn.W2.shape)

        # Print results
        tribe_count = len(set(c.tribe_id for c in self.creatures if c.tribe_id is not None))
        
        print(f"\n--- Generation {self.currentgeneration} Results ---")
        print(f"Fittest Score: {fittest.fitness:.4f}")
        print(f"Tokens: B:{fittest.brain_tokens}, D:{fittest.body_tokens}, L:{fittest.leg_tokens}")
        print(f"Leg Directions: {fittest.leg_sub_tokens}")
        print(f"Tribes Formed: {tribe_count}")
        print(f"Fittest's Speech Tokens: {fittest.speech_tokens}")
        
        self.current_generation += 1
        return fittest.fitness

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # Initialize the environment
    env = Environment(world_size=100, num_creatures=100)
    
    # Run 20 generations
    num_generations = 20
    
    for _ in range(num_generations):
        best_fitness = env.run_generation(time_steps=200)

        # Check for convergence (perfect fitness)
        if best_fitness >= 1.0:
            print("\n!!! GOAL ACHIEVED: A creature reached perfect fitness (Golden Ratio) !!!")
            break