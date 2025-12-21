"""
Improved TD3 Parking Allocation Agent
Removed global variable, added proper error handling and logging.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import List, Optional, Dict, Any
import logging
import config

logger = logging.getLogger(__name__)

# --- 1. NEURAL NETWORK (Must match train_simulation.py) ---
class Actor(nn.Module):
    """Policy network for parking spot selection."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return torch.tanh(self.l3(x))  # Output in [-1, 1]

# --- 2. TD3 AGENT CLASS ---
class TD3ParkingAgent:
    """
    TD3-based intelligent parking spot allocation agent.
    Uses trained neural network to select optimal parking spots.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        state_dim: int = None,
        action_dim: int = None,
        hidden_dim: int = None
    ):
        """
        Initialize TD3 Parking Agent.
        
        Args:
            model_path: Path to saved model weights
            state_dim: State dimension (defaults to config)
            action_dim: Action dimension (defaults to config)
            hidden_dim: Hidden layer size (defaults to config)
        """
        # Use config defaults if not specified
        self.state_dim = state_dim or config.TD3_STATE_DIM
        self.action_dim = action_dim or config.TD3_ACTION_DIM
        self.hidden_dim = hidden_dim or config.TD3_HIDDEN_DIM
        self.model_path = model_path or config.TD3_MODEL_PATH
        
        # Initialize network
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_dim)
        self.is_trained = False
        
        # Try to load trained model
        self._load_model()
    
    def _load_model(self) -> bool:
        """
        Load trained model weights.
        
        Returns:
            True if successfully loaded, False otherwise
        """
        if not os.path.exists(self.model_path):
            logger.warning(f"⚠️ Model file not found: {self.model_path}")
            logger.warning("Agent will use untrained (random) weights!")
            print(f"⚠️ No trained model found at '{self.model_path}'")
            print("   Run train_simulation.py first to train the agent.")
            return False
        
        try:
            # Load model (ensure CPU compatibility for Raspberry Pi)
            state_dict = torch.load(
                self.model_path,
                map_location=torch.device('cpu'),
                weights_only=True  # Security: only load weights
            )
            self.actor.load_state_dict(state_dict)
            self.actor.eval()  # Set to evaluation mode
            
            self.is_trained = True
            logger.info(f"✅ TD3 model loaded from: {self.model_path}")
            print("🧠 TD3 Brain Loaded: Using Trained Intelligence")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            print(f"⚠️ Error loading model: {e}")
            print("   Agent will use untrained weights.")
            return False
    
    def _decode_action(self, action_value: float) -> int:
        """
        Decode continuous action to discrete spot index.
        
        Args:
            action_value: Continuous value in [-1, 1]
            
        Returns:
            Spot index (0, 1, or 2)
        """
        if action_value < config.ACTION_THRESHOLD_LOW:
            return 0  # Spot 1
        elif action_value > config.ACTION_THRESHOLD_HIGH:
            return 2  # Spot 3
        else:
            return 1  # Spot 2
    
    def _validate_inputs(
        self,
        sensor_states: List[int],
        building_choice: str
    ) -> bool:
        """
        Validate input parameters.
        
        Args:
            sensor_states: List of sensor readings
            building_choice: Building identifier
            
        Returns:
            True if valid, False otherwise
        """
        # Check sensor states
        if not isinstance(sensor_states, list):
            logger.error("sensor_states must be a list")
            return False
        
        if len(sensor_states) != len(config.PARKING_SPOTS):
            logger.error(f"Expected {len(config.PARKING_SPOTS)} sensors, got {len(sensor_states)}")
            return False
        
        if not all(s in [0, 1] for s in sensor_states):
            logger.error("Sensor states must be 0 or 1")
            return False
        
        # Check building choice
        if building_choice not in config.BUILDINGS:
            logger.error(f"Invalid building: {building_choice}. Must be one of {list(config.BUILDINGS.keys())}")
            return False
        
        return True
    
    def select_spot(
        self,
        sensor_states: List[int],
        building_choice: str,
        use_ai: bool = True
    ) -> Dict[str, Any]:
        """
        Select optimal parking spot based on sensor data and destination.
        
        Args:
            sensor_states: List [1, 0, 1] where 1=Free, 0=Occupied
            building_choice: "A" or "B" - destination building
            use_ai: If False, uses fallback heuristic only
            
        Returns:
            Dictionary with:
                - spot_id: Selected spot ID (1-3) or -1 if full
                - method: 'ai', 'fallback', or 'full'
                - confidence: AI confidence score (if applicable)
                - action_value: Raw AI output (if applicable)
        """
        # Validate inputs
        if not self._validate_inputs(sensor_states, building_choice):
            logger.error("Invalid inputs to select_spot")
            return {"spot_id": -1, "method": "error", "confidence": 0.0}
        
        # Check if lot is completely full
        if sum(sensor_states) == 0:
            logger.info("Parking lot is full")
            return {"spot_id": -1, "method": "full", "confidence": 1.0}
        
        # AI-based selection
        if use_ai and self.is_trained:
            try:
                # Prepare state vector
                dest_loc = config.BUILDINGS[building_choice]
                state_vector = sensor_states + [dest_loc]
                state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
                
                # Get AI prediction
                with torch.no_grad():
                    action_value = self.actor(state_tensor).item()
                
                # Decode to spot index
                suggested_idx = self._decode_action(action_value)
                
                # Verify the AI's choice is actually free
                if sensor_states[suggested_idx] == 1:
                    spot_id = config.PARKING_SPOTS[suggested_idx]["id"]
                    logger.info(f"AI selected spot {spot_id} (action={action_value:.3f})")
                    
                    return {
                        "spot_id": spot_id,
                        "method": "ai",
                        "confidence": 0.8,  # Could be computed from Q-values
                        "action_value": action_value
                    }
                else:
                    logger.warning(f"AI suggested occupied spot {suggested_idx + 1}, using fallback")
                    
            except Exception as e:
                logger.error(f"AI selection failed: {e}", exc_info=True)
        
        # Fallback: Greedy nearest-neighbor
        return self._fallback_selection(sensor_states, building_choice)
    
    def _fallback_selection(
        self,
        sensor_states: List[int],
        building_choice: str
    ) -> Dict[str, Any]:
        """
        Fallback heuristic: select nearest free spot to destination.
        
        Args:
            sensor_states: Sensor readings
            building_choice: Destination building
            
        Returns:
            Selection result dictionary
        """
        dest_loc = config.BUILDINGS[building_choice]
        best_spot_id = -1
        min_distance = float('inf')
        
        for idx, is_free in enumerate(sensor_states):
            if is_free == 1:  # Spot is available
                spot_loc = config.PARKING_SPOTS[idx]["location"]
                distance = abs(spot_loc - dest_loc)
                
                if distance < min_distance:
                    min_distance = distance
                    best_spot_id = config.PARKING_SPOTS[idx]["id"]
        
        if best_spot_id != -1:
            logger.info(f"Fallback selected spot {best_spot_id} (distance={min_distance:.1f})")
        
        return {
            "spot_id": best_spot_id,
            "method": "fallback",
            "confidence": 0.6,
            "distance": min_distance
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get agent status information.
        
        Returns:
            Status dictionary
        """
        return {
            "is_trained": self.is_trained,
            "model_path": self.model_path,
            "model_exists": os.path.exists(self.model_path),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim
        }

# --- 3. FACTORY FUNCTION (Replaces global variable) ---
_agent_instance: Optional[TD3ParkingAgent] = None

def get_agent() -> TD3ParkingAgent:
    """
    Get or create the global TD3 agent instance (singleton pattern).
    This replaces the old global 'brain' variable with lazy initialization.
    
    Returns:
        TD3ParkingAgent instance
    """
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = TD3ParkingAgent()
    return _agent_instance

# --- Backward compatibility: 'brain' for existing code ---
# This allows old code to still work, but new code should use get_agent()
brain = None  # Will be initialized on first access via property

def __getattr__(name: str):
    """Module-level attribute access for backward compatibility."""
    if name == "brain":
        return get_agent()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# --- Test functionality ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing TD3 Parking Agent...\n")
    
    # Create agent
    agent = TD3ParkingAgent()
    
    # Print status
    status = agent.get_status()
    print("Agent Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Test scenarios
    print("\n--- Test Scenarios ---")
    
    test_cases = [
        ([1, 1, 1], "A", "All spots free, going to A"),
        ([1, 1, 1], "B", "All spots free, going to B"),
        ([0, 1, 1], "A", "Spot 1 occupied, going to A"),
        ([1, 1, 0], "B", "Spot 3 occupied, going to B"),
        ([0, 0, 1], "A", "Only spot 3 free"),
        ([0, 0, 0], "A", "All spots occupied"),
    ]
    
    for sensors, dest, description in test_cases:
        result = agent.select_spot(sensors, dest)
        print(f"\n{description}")
        print(f"  Sensors: {sensors}, Destination: {dest}")
        print(f"  Result: Spot {result['spot_id']}, Method: {result['method']}")