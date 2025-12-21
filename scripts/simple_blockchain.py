"""
Improved Blockchain Implementation with Validation and Persistence Loading
"""
import hashlib
import json
import time
import copy
import os
from typing import Dict, List, Optional, Any
import logging
import config

logger = logging.getLogger(__name__)

class Block:
    """
    Represents a single block in the blockchain.
    """
    def __init__(
        self,
        index: int,
        previous_hash: str,
        timestamp: float,
        data: Dict[str, Any],
        nonce: int = 0
    ):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.nonce = nonce
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        """
        Calculates the SHA-256 hash of the block.
        Excludes the 'hash' field itself to prevent circular dependency.
        """
        # Create a copy to avoid modifying the original
        block_copy = copy.deepcopy(self.__dict__)
        
        # Remove the hash field from calculation
        if "hash" in block_copy:
            del block_copy["hash"]
            
        # Convert to JSON and hash
        block_string = json.dumps(block_copy, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary for serialization."""
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Block':
        """Create block from dictionary."""
        block = cls(
            index=data['index'],
            previous_hash=data['previous_hash'],
            timestamp=data['timestamp'],
            data=data['data'],
            nonce=data.get('nonce', 0)
        )
        # Use the stored hash instead of recalculating
        block.hash = data['hash']
        return block

class Blockchain:
    """
    Blockchain for secure, tamper-proof vehicle entry logging.
    """
    def __init__(self, auto_load: bool = True):
        """
        Initialize blockchain and optionally load from disk.
        
        Args:
            auto_load: If True, attempts to load existing chain from disk
        """
        self.chain: List[Block] = []
        self.blockchain_file = config.BLOCKCHAIN_FILE
        
        if auto_load and os.path.exists(self.blockchain_file):
            if not self.load_chain():
                logger.warning("Failed to load blockchain, creating new genesis block")
                self.create_genesis_block()
        else:
            self.create_genesis_block()

    def create_genesis_block(self) -> None:
        """Creates the initial block in the blockchain."""
        genesis_data = {
            "info": "Smart Parking System - Genesis Block",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0"
        }
        genesis_block = Block(0, "0", time.time(), genesis_data)
        self.chain.append(genesis_block)
        self.save_chain()
        logger.info("🔗 Genesis block created")

    def get_latest_block(self) -> Block:
        """Returns the most recent block in the chain."""
        return self.chain[-1]

    def add_block(self, data: Dict[str, Any]) -> Optional[Block]:
        """
        Adds a new block to the blockchain.
        
        Args:
            data: Dictionary containing block data
            
        Returns:
            New block if successful, None otherwise
        """
        try:
            last_block = self.get_latest_block()
            
            new_block = Block(
                index=last_block.index + 1,
                previous_hash=last_block.hash,
                timestamp=time.time(),
                data=data
            )
            
            self.chain.append(new_block)
            self.save_chain()
            
            logger.info(f"🔗 Block #{new_block.index} added [Hash: {new_block.hash[:12]}...]")
            return new_block
            
        except Exception as e:
            logger.error(f"Failed to add block: {e}")
            return None

    def is_chain_valid(self) -> bool:
        """
        Validates the entire blockchain for integrity.
        Checks both hash consistency and chain linkage.
        
        Returns:
            True if blockchain is valid, False otherwise
        """
        try:
            for i in range(1, len(self.chain)):
                current = self.chain[i]
                previous = self.chain[i-1]

                # Check 1: Verify block's internal hash integrity
                recalculated_hash = current.calculate_hash()
                if current.hash != recalculated_hash:
                    logger.error(f"❌ Block #{current.index} hash corrupted!")
                    logger.error(f"   Stored: {current.hash[:16]}...")
                    logger.error(f"   Calculated: {recalculated_hash[:16]}...")
                    return False
                
                # Check 2: Verify link to previous block
                if current.previous_hash != previous.hash:
                    logger.error(f"❌ Block #{current.index} chain link broken!")
                    logger.error(f"   Expected: {previous.hash[:16]}...")
                    logger.error(f"   Got: {current.previous_hash[:16]}...")
                    return False
            
            logger.info(f"✅ Blockchain validated: {len(self.chain)} blocks verified")
            return True
            
        except Exception as e:
            logger.error(f"Blockchain validation error: {e}")
            return False

    def save_chain(self) -> bool:
        """
        Saves the blockchain to a JSON file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            chain_data = [block.to_dict() for block in self.chain]
            
            # Write to temporary file first
            temp_file = f"{self.blockchain_file}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(chain_data, f, indent=4)
            
            # Atomic rename
            os.replace(temp_file, self.blockchain_file)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save blockchain: {e}")
            return False

    def load_chain(self) -> bool:
        """
        Loads the blockchain from a JSON file and validates it.
        
        Returns:
            True if successfully loaded and valid, False otherwise
        """
        try:
            if not os.path.exists(self.blockchain_file):
                logger.info("No existing blockchain file found")
                return False
            
            with open(self.blockchain_file, 'r', encoding='utf-8') as f:
                chain_data = json.load(f)
            
            if not chain_data:
                logger.warning("Empty blockchain file")
                return False
            
            # Reconstruct blocks
            self.chain = [Block.from_dict(block_data) for block_data in chain_data]
            
            # Validate the loaded chain
            if not self.is_chain_valid():
                logger.error("⚠️ Loaded blockchain failed validation!")
                self.chain = []
                return False
            
            logger.info(f"✅ Blockchain loaded: {len(self.chain)} blocks")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid blockchain JSON: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load blockchain: {e}")
            return False
    
    def get_block_by_index(self, index: int) -> Optional[Block]:
        """Get a specific block by its index."""
        if 0 <= index < len(self.chain):
            return self.chain[index]
        return None
    
    def search_blocks(self, key: str, value: Any) -> List[Block]:
        """
        Search for blocks containing specific data.
        
        Args:
            key: Key to search for in block data
            value: Value to match
            
        Returns:
            List of matching blocks
        """
        matching_blocks = []
        for block in self.chain[1:]:  # Skip genesis block
            if key in block.data and block.data[key] == value:
                matching_blocks.append(block)
        return matching_blocks
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get blockchain statistics."""
        return {
            'total_blocks': len(self.chain),
            'genesis_time': self.chain[0].timestamp if self.chain else None,
            'latest_time': self.chain[-1].timestamp if self.chain else None,
            'file_size_bytes': os.path.getsize(self.blockchain_file) if os.path.exists(self.blockchain_file) else 0
        }

# Global Instance - initialized with auto-loading
ledger = Blockchain(auto_load=True)

# Test and validation on import
if __name__ != "__main__":
    # Validate blockchain on import (production use)
    if not ledger.is_chain_valid():
        logger.critical("⚠️ BLOCKCHAIN INTEGRITY COMPROMISED!")
        
# --- Test functionality ---
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Blockchain Implementation...")
    
    # Create test blockchain
    test_chain = Blockchain(auto_load=False)
    
    # Add some test blocks
    test_chain.add_block({"plate": "ABC 123", "spot": 1})
    test_chain.add_block({"plate": "XYZ 789", "spot": 2})
    test_chain.add_block({"plate": "DEF 456", "spot": 3})
    
    # Validate
    print(f"\nValidation result: {test_chain.is_chain_valid()}")
    
    # Test tampering detection
    print("\n--- Testing Tampering Detection ---")
    test_chain.chain[1].data["plate"] = "HACKED"
    print(f"After tampering: {test_chain.is_chain_valid()}")
    
    # Statistics
    print(f"\nStatistics: {test_chain.get_statistics()}")