import simple_blockchain
import json
import os

def load_chain_from_file():
    filename = "secure_ledger.json"
    
    if not os.path.exists(filename):
        print("❌ No blockchain file found. Run the logger first.")
        return False

    # Load the JSON file
    with open(filename, "r") as f:
        data = json.load(f)

    # Reconstruct the Blockchain object from the file data
    # We clear the current memory and rebuild it block by block
    simple_blockchain.ledger.chain = []
    
    print(f"📂 Loaded {len(data)} blocks from {filename}")
    
    for block_data in data:
        # Recreate Block Object
        blk = simple_blockchain.Block(
            block_data["index"],
            block_data["previous_hash"],
            block_data["timestamp"],
            block_data["data"],
            block_data["nonce"]
        )
        # IMPORTANT: We must manually set the hash to what was in the file
        # so we can check if it matches the calculation later
        blk.hash = block_data["hash"]
        
        simple_blockchain.ledger.chain.append(blk)
    
    return True

def tamper_test():
    """
    OPTIONAL: Simulates a hacker changing data to prove the system works.
    """
    if len(simple_blockchain.ledger.chain) > 1:
        print("\n😈 SIMULATING HACK ATTACK...")
        # Hacker changes the plate number in Block 1
        target_block = simple_blockchain.ledger.chain[1]
        original_data = target_block.data["plate"]
        
        target_block.data["plate"] = "HACKED-999"
        print(f"   Changed Block #1 Data: '{original_data}' -> 'HACKED-999'")
        
        print("🔍 Re-running Verification...")
        if simple_blockchain.ledger.is_chain_valid():
            print("❌ TEST FAILED: System did not detect the hack!")
        else:
            print("✅ TEST PASSED: System successfully detected the hack!")

def main():
    print("🛡️ BLOCKCHAIN SECURITY AUDIT 🛡️")
    print("--------------------------------")
    
    # 1. Load Data
    if load_chain_from_file():
        
        # 2. Verify Integrity
        print("\n🔍 Verifying Chain Integrity...")
        if simple_blockchain.ledger.is_chain_valid():
            print("✅ STATUS: SECURE. No tampering detected.")
            print("   All hashes match their data.")
        else:
            print("❌ STATUS: COMPROMISED! Data has been altered.")

        # 3. Optional: Run a fake hack to demonstrate
        # Uncomment the line below if you want to show the professor "What if..."
        # tamper_test()

if __name__ == "__main__":
    main()