"""
Chapter 2 Hands-On Lab: Python Collections & The ApexPay FinTech Case Study
Modern Machine Learning & Data Science Engineering
Author: Vaibhav • Enterprise Data Architect

Designed for beginners with zero prior programming experience.
Covers:
  1. Lists: Indexing, Slicing, Mutability & Essential Methods
  2. Tuples: Immutability, Waypoint Coordinates & Tuple Unpacking
  3. Sets: Deduplication & Mathematical Set Operations (CyberShield Firewall)
  4. Dictionaries: Key-Value Lookups, Defensive .get(), and Nested Profiles
  5. Comprehensions: 1-Line Data Filtering & Dictionary Transformations
  6. Advanced Functions: *args, **kwargs, Lambda Functions, map() and filter()
  7. End-to-End Applied Case Study: "ApexPay" Merchant Transaction Risk Engine
"""
import pandas as pd

# ==============================================================================
# SECTION 1: LISTS IN DEPTH (AEROWINGS FLIGHT MANIFEST)
# ==============================================================================
def demo_section_1():
    print("=" * 70)
    print("SECTION 1: LISTS, SLICING & MUTABLE OPERATIONS")
    print("=" * 70)

    # Creating a list of passenger names
    passengers = ["Sarah Connor", "John Doe", "Alice Smith", "Bob Vance", "Elena Rostova"]
    print(f"Original Manifest ({len(passengers)} passengers): {passengers}")

    # Zero-based indexing and negative indexing
    first_passenger = passengers[0]
    last_passenger = passengers[-1]
    second_to_last = passengers[-2]
    print(f"First (index 0): {first_passenger} | Last (index -1): {last_passenger} | Penultimate (-2): {second_to_last}")

    # Slicing: [start:stop:step] -> stop is non-inclusive!
    first_three = passengers[:3]
    middle_two = passengers[1:3]
    alternate_passengers = passengers[::2]
    reversed_manifest = passengers[::-1]
    print(f"First 3 (:3):       {first_three}")
    print(f"Middle [1:3]:       {middle_two}")
    print(f"Alternate (::2):    {alternate_passengers}")
    print(f"Reversed (::-1):    {reversed_manifest}")

    # Mutability: Modifying an element in place
    passengers[1] = "Jonathan Doe, MD"
    print(f"\nUpdated Name at Index 1: {passengers[1]}")

    # Essential List Methods
    passengers.append("Marcus Aurelius")      # Adds to end
    passengers.insert(0, "VIP: Captain Smith") # Inserts at index 0
    popped_passenger = passengers.pop()       # Removes and returns last element
    passengers.remove("Alice Smith")          # Removes specific value

    print(f"After Append, Insert, Pop ({popped_passenger}), and Remove: {passengers}")


# ==============================================================================
# SECTION 2: TUPLES & IMMUTABILITY (GPS WAYPOINTS & GEO COORDINATES)
# ==============================================================================
def demo_section_2():
    print("\n" + "=" * 70)
    print("SECTION 2: TUPLES & THE POWER OF IMMUTABILITY")
    print("=" * 70)

    # Tuples are ordered collections enclosed in parentheses ()
    # Key property: IMMUTABLE (cannot be modified after creation)
    warehouse_hq = (37.7749, -122.4194, "San Francisco Hub")
    delivery_dropoff = (37.3382, -121.8863, "San Jose Client")

    # Tuple Unpacking
    lat, lon, label = warehouse_hq
    print(f"HQ Latitude: {lat} | Longitude: {lon} | Location: {label}")

    # Demonstrating immutability protection
    print("\nAttempting to reassign warehouse_hq[0] = 38.0000...")
    try:
        warehouse_hq[0] = 38.0000
    except TypeError as e:
        print(f"  [PROTECTION TRIGGERED] TypeError: {e}")
        print("  -> Tuples guarantee that critical coordinates cannot be accidentally overwritten!")


# ==============================================================================
# SECTION 3: SETS & MATHEMATICAL OPERATIONS (CYBERSHIELD FIREWALL AUDIT)
# ==============================================================================
def demo_section_3():
    print("\n" + "=" * 70)
    print("SECTION 3: SETS & HIGH-PERFORMANCE DEDUPLICATION")
    print("=" * 70)

    # Raw server logs containing duplicate IP addresses
    raw_incoming_ips = [
        "192.168.1.10", "10.0.0.5", "192.168.1.10", "172.16.0.4",
        "10.0.0.5", "192.168.1.99", "10.0.0.5", "192.168.1.10"
    ]

    # Deduplicating using a set
    unique_visitors = set(raw_incoming_ips)
    print(f"Raw Log Count:    {len(raw_incoming_ips)} requests")
    print(f"Unique IP Count:  {len(unique_visitors)} distinct hosts -> {unique_visitors}")

    # Global threat intelligence blacklist set
    known_threat_ips = {"192.168.1.99", "203.0.113.5", "198.51.100.22", "172.16.0.4"}

    # Set Operations:
    # 1. Intersection (&): Incoming IPs that appear on the threat blacklist
    active_threats = unique_visitors & known_threat_ips
    # 2. Difference (-): Safe unique visitors not on the blacklist
    safe_visitors = unique_visitors - known_threat_ips
    # 3. Union (|): Complete combined universe of observed + known threat IPs
    all_monitored_ips = unique_visitors | known_threat_ips

    print(f"\n[BLOCKED] Active Threats Identified (&): {active_threats}")
    print(f"[ALLOWED] Clean Verified Visitors (-):   {safe_visitors}")
    print(f"Total Monitored Threat Universe (|):      {len(all_monitored_ips)} total IPs")


# ==============================================================================
# SECTION 4: DICTIONARIES & DEFENSIVE ACCESS (CUSTOMER 360 PROFILES)
# ==============================================================================
def demo_section_4():
    print("\n" + "=" * 70)
    print("SECTION 4: DICTIONARIES & NESTED RECORD LOOKUPS")
    print("=" * 70)

    # Dictionaries store data as key-value pairs (hash maps)
    client_profile = {
        "account_id": "ACC-8831",
        "name": "Evelyn Reed",
        "tier": "Gold",
        "lifetime_spend": 14250.75,
        "active_contracts": ["Enterprise_SaaS", "Cloud_Backup"],
        "metadata": {
            "industry": "Healthcare",
            "region": "North America",
            "risk_score": 0.12
        }
    }

    print(f"Client: {client_profile['name']} (Tier: {client_profile['tier']})")
    print(f"Industry: {client_profile['metadata']['industry']}")

    # The Dangerous Way: Direct key indexing crashes if the key does not exist
    print("\nLooking up optional key 'referral_code'...")
    try:
        print(client_profile["referral_code"])
    except KeyError as e:
        print(f"  [CRASH AVERTED] KeyError: {e}")

    # The Professional Way: .get() returns a friendly fallback default!
    referral = client_profile.get("referral_code", "NO_REFERRAL_CODE")
    churn_flag = client_profile.get("is_churned", False)
    print(f"Safe Lookup with .get(): referral={referral}, is_churned={churn_flag}")

    # Iterating over key-value pairs with .items()
    print("\nTop-Level Profile Summary:")
    for key, val in client_profile.items():
        if key != "metadata" and key != "active_contracts":
            print(f"  {key:<18}: {val}")


# ==============================================================================
# SECTION 5: COMPREHENSIONS (FAST DATA FILTERING & TRANSFORMATION)
# ==============================================================================
def demo_section_5():
    print("\n" + "=" * 70)
    print("SECTION 5: LIST & DICTIONARY COMPREHENSIONS")
    print("=" * 70)

    transactions = [12.50, 45.00, 120.00, 8.99, 65.40, 210.00, 34.00, 500.00]

    # Traditional loop (4 lines) vs List Comprehension (1 line!)
    # Keep only high-value transactions (>= $50.00) and apply 10% loyalty rebate
    high_value_rebates = [round(t * 0.90, 2) for t in transactions if t >= 50.00]
    print(f"Original Transactions:      {transactions}")
    print(f"Filtered & Rebated (>= $50): {high_value_rebates}")

    # Dictionary Comprehension: Converting currencies (USD to EUR @ 0.92 exchange rate)
    usd_prices = {"Laptop": 1200, "Monitor": 350, "Keyboard": 90, "Mouse": 45}
    eur_prices = {item: round(usd * 0.92, 2) for item, usd in usd_prices.items()}
    print(f"\nUSD Catalog: {usd_prices}")
    print(f"EUR Catalog: {eur_prices}")


# ==============================================================================
# SECTION 6: ADVANCED FUNCTIONS (*ARGS, **KWARGS & LAMBDAS)
# ==============================================================================
def calculate_invoice_total(base_amount: float, *fees, **discounts) -> float:
    """
    Demonstrates *args (arbitrary positional fees) and **kwargs (arbitrary keyword discounts).
    """
    total = base_amount + sum(fees)
    total_discount = sum(discounts.values())
    return max(0.0, total - total_discount)

def demo_section_6():
    print("\n" + "=" * 70)
    print("SECTION 6: *ARGS, **KWARGS & LAMBDA FUNCTIONS")
    print("=" * 70)

    # Calling with arbitrary fees (*args) and named discounts (**kwargs)
    invoice = calculate_invoice_total(
        1000.00,           # base_amount
        50.00, 25.00, 15.00,# *fees (shipping, handling, tax)
        promo=100.00, vip=50.00 # **discounts
    )
    print(f"Invoice Total with Dynamic Fees & Discounts: ${invoice:.2f}")

    # Lambda functions with map() and filter()
    temperatures_c = [18.5, 22.0, 31.5, 14.0, 27.5]
    # Convert Celsius to Fahrenheit: (C * 9/5) + 32
    to_fahr = lambda c: round((c * 9/5) + 32, 1)
    temperatures_f = list(map(to_fahr, temperatures_c))
    print(f"Celsius Readings:    {temperatures_c}")
    print(f"Mapped Fahrenheit:   {temperatures_f}")

    # Filter hot days (> 80°F)
    hot_days = list(filter(lambda f: f > 80.0, temperatures_f))
    print(f"Filtered Hot Days:   {hot_days}")


# ==============================================================================
# SECTION 7: END-TO-END FINTECH CASE STUDY: "APEXPAY" RISK ENGINE
# ==============================================================================
def run_apexpay_case_study():
    """
    Complete progressive case study mimicking real-world data engineering:
      Step 1: Storing customer transactions as naive separate variables
      Step 2: Organizing into parallel lists & querying with .index()
      Step 3: Demonstrating the fragility of parallel lists
      Step 4: Re-architecting into a structured database of dictionaries
      Step 5: Auditing, filtering, and risk scoring transactions
      Step 6: Writing modular update functions with defensive error handling
      Step 7: Converting into a structured Pandas DataFrame for reporting
    """
    print("\n" + "=" * 70)
    print("SECTION 7: COMPLETE CASE STUDY - 'APEXPAY' TRANSACTION RISK ENGINE")
    print("=" * 70)

    # --- STEP 1: The Naive Way (Separate Variables) ---
    print("--- Step 1: Why Separate Variables Fail at Scale ---")
    tx1_id, tx1_amount, tx1_country, tx1_is_fraud = "TX-101", 150.00, "US", False
    tx2_id, tx2_amount, tx2_country, tx2_is_fraud = "TX-102", 4800.00, "RU", False
    print(f"Transaction 1: {tx1_id}, ${tx1_amount}, {tx1_country}")
    print(f"Transaction 2: {tx2_id}, ${tx2_amount}, {tx2_country}")
    print(">> Problem: If we have 10,000 transactions, we cannot create 40,000 separate variables!\n")

    # --- STEP 2 & 3: Parallel Lists & Their Dangerous Fragility ---
    print("--- Step 2 & 3: Grouping into Lists & Why Parallel Lists are Fragile ---")
    tx_ids = ["TX-101", "TX-102", "TX-103", "TX-104", "TX-105"]
    amounts = [150.00, 4800.00, 32.50, 9200.00, 210.00]
    countries = ["US", "RU", "US", "NG", "GB"]
    fraud_flags = [False, False, False, False, False]

    # Querying transaction TX-102
    idx = tx_ids.index("TX-102")
    print(f"Looked up TX-102 at index {idx}: Amount=${amounts[idx]}, Country={countries[idx]}")

    # THE DANGER: If someone deletes or inserts an element in just ONE list:
    # amounts.pop(1) -> All subsequent indices are now permanently desynchronized!
    print(">> Fragility Warning: If one list is sorted or trimmed independently, customer IDs")
    print("   no longer align with amounts or countries. Silent data corruption occurs!\n")

    # --- STEP 4: The Clean Architectural Solution: List of Dictionaries ---
    print("--- Step 4: Re-architecting into a Database of Dictionaries ---")
    transactions_db = [
        {"tx_id": "TX-101", "merchant": "Metro Grocers", "amount": 150.00,  "country": "US", "risk_score": 0.05, "status": "APPROVED"},
        {"tx_id": "TX-102", "merchant": "Global Crypto",  "amount": 4800.00, "country": "RU", "risk_score": 0.65, "status": "REVIEW"},
        {"tx_id": "TX-103", "merchant": "Coffee Haven",   "amount": 32.50,   "country": "US", "risk_score": 0.01, "status": "APPROVED"},
        {"tx_id": "TX-104", "merchant": "Wire Transfer",  "amount": 9200.00, "country": "NG", "risk_score": 0.88, "status": "REVIEW"},
        {"tx_id": "TX-105", "merchant": "London Books",   "amount": 210.00,  "country": "GB", "risk_score": 0.10, "status": "APPROVED"},
    ]
    print(f"Constructed database with {len(transactions_db)} unified transaction objects.")

    # --- STEP 5: Automated Risk Auditing Logic ---
    print("\n--- Step 5: Auditing Suspicious Transactions ---")
    high_risk_countries = {"RU", "NG", "KP", "IR"}
    flagged_for_investigation = []

    for tx in transactions_db:
        # Rule 1: High transaction amount
        # Rule 2: Originates from high-risk jurisdiction
        # Rule 3: High computed risk score
        is_suspicious = (tx["amount"] > 3000.00) or (tx["country"] in high_risk_countries) or (tx["risk_score"] > 0.70)
        if is_suspicious:
            tx["status"] = "FLAGGED_FOR_AUDIT"
            flagged_for_investigation.append(tx)

    print(f">> Total Flagged Transactions: {len(flagged_for_investigation)}")
    for f in flagged_for_investigation:
        print(f"   [ALERT] {f['tx_id']}: ${f['amount']:>8,.2f} from {f['country']} ({f['merchant']}) -> Status: {f['status']}")

    # --- STEP 6: Modular Transaction Update Function ---
    def update_transaction_status(db: list, target_id: str, new_status: str) -> bool:
        """Safely updates a transaction status by ID with defensive validation."""
        valid_statuses = {"APPROVED", "FLAGGED_FOR_AUDIT", "SETTLED", "REJECTED"}
        if new_status not in valid_statuses:
            raise ValueError(f"Invalid status '{new_status}'. Allowed: {valid_statuses}")

        for record in db:
            if record.get("tx_id") == target_id:
                record["status"] = new_status
                return True
        return False

    update_transaction_status(transactions_db, "TX-104", "REJECTED")
    print(f"\nUpdated TX-104 status: {[t['status'] for t in transactions_db if t['tx_id'] == 'TX-104'][0]}")

    # --- STEP 7: Exporting to Pandas DataFrame ---
    print("\n--- Step 7: Converting Structured Dictionary Data to Pandas DataFrame ---")
    df = pd.DataFrame(transactions_db)
    print(df[["tx_id", "merchant", "amount", "country", "risk_score", "status"]])
    print(f"\nTotal Portfolio Processed: ${df['amount'].sum():,.2f}")
    print(f"Average Transaction Size:  ${df['amount'].mean():,.2f}")
    print("\n[SUCCESS] ApexPay Case Study Completed Successfully!")


if __name__ == "__main__":
    demo_section_1()
    demo_section_2()
    demo_section_3()
    demo_section_4()
    demo_section_5()
    demo_section_6()
    run_apexpay_case_study()
