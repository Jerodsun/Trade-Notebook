from greeks import calculate_iv, calculate_greeks
from db_manager import create_position, add_execution, get_open_positions, get_executions, delete_position

def test_greeks():
    print("Testing Greeks Calculation...")
    S = 5000 # Underlying price
    K = 5000 # Strike price
    T = 30 / 365 # 30 days to expiry
    r = 0.045 # 4.5% rate
    market_price = 100 # Option price
    
    iv = calculate_iv(S, K, T, r, market_price, 'Call')
    print(f"Calculated IV: {iv:.4f}")
    
    greeks = calculate_greeks(S, K, T, r, iv, 'Call')
    print(f"Greeks: {greeks}")
    assert greeks['delta'] > 0
    assert greeks['gamma'] > 0
    print("Greeks test passed!\n")

def test_db():
    print("Testing Database CRUD...")
    # Create a test position
    pos_id = create_position("TEST", 5000, "2026-06-13", "Call", "Test notes")
    print(f"Created position ID: {pos_id}")
    
    # Add an execution (Buy)
    add_execution(pos_id, "Buy", 1, 100, 5000)
    print("Added Buy execution")
    
    # Check open positions
    open_pos = get_open_positions()
    assert any(p['id'] == pos_id for p in open_pos)
    print("Position is in Open list")
    
    # Add an execution (Sell - to close)
    add_execution(pos_id, "Sell", 1, 120, 5100)
    print("Added Sell execution")
    
    # Check executions
    execs = get_executions(pos_id)
    assert len(execs) == 2
    print(f"Found {len(execs)} executions")
    
    # Check if closed
    open_pos = get_open_positions()
    assert not any(p['id'] == pos_id for p in open_pos)
    print("Position successfully closed")

    # Delete position
    delete_position(pos_id)
    execs = get_executions(pos_id)
    assert len(execs) == 0
    print("Position and executions successfully deleted")
    print("Database test passed!")

if __name__ == "__main__":
    test_greeks()
    test_db()
