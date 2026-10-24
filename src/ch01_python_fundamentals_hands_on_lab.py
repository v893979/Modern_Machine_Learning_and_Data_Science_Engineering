"""
Chapter 1 Hands-On Lab: Python Fundamentals from Scratch
Modern Machine Learning & Data Science Engineering
Author: Vaibhav • Enterprise Data Architect

Designed for beginners with zero prior programming experience.
Covers:
  1. Variables, Data Types & Dynamic Typing
  2. Type Casting & User Input Simulation (Patient Health Triage)
  3. Arithmetic, Comparison & Logical Operators (Food Delivery Fee Engine)
  4. Decision Making with if / elif / else (Credit Card Fraud Alert)
  5. While Loops & Sentinel Variables (EV Fleet Battery Drain Simulation)
  6. For Loops, range(), enumerate(), and zip() (Store Inventory Audit)
  7. Modular Functions, Default Parameters & Multiple Return Values (Loan DTI Calculator)
"""

# ==============================================================================
# SECTION 1: VARIABLES, DATA TYPES & DYNAMIC TYPING
# ==============================================================================
def demo_section_1():
    print("=" * 70)
    print("SECTION 1: VARIABLES & DATA TYPES")
    print("=" * 70)

    # In Python, variables are named labels pointing to data stored in memory
    patient_id = "P-10492"           # String (text)
    patient_age = 42                 # Integer (whole number)
    body_temperature = 98.6          # Float (decimal number)
    has_preexisting_condition = True # Boolean (True or False)

    print(f"Patient ID:    {patient_id} (Type: {type(patient_id).__name__})")
    print(f"Age:           {patient_age} (Type: {type(patient_age).__name__})")
    print(f"Temperature:   {body_temperature}°F (Type: {type(body_temperature).__name__})")
    print(f"Pre-existing:  {has_preexisting_condition} (Type: {type(has_preexisting_condition).__name__})")

    # Dynamic typing: A variable can change its type when reassigned
    status = "Pending Triage"
    print(f"\nInitial Status: {status} ({type(status).__name__})")
    status = 1  # 1 = Admitted, 0 = Discharged
    print(f"Updated Status: {status} ({type(status).__name__})")

    # Multiple assignment
    systolic, diastolic = 120, 80
    print(f"Blood Pressure: {systolic}/{diastolic} mmHg")


# ==============================================================================
# SECTION 2: TYPE CASTING & FORMULA EVALUATION (PATIENT HEALTH TRIAGE)
# ==============================================================================
def demo_section_2():
    print("\n" + "=" * 70)
    print("SECTION 2: TYPE CASTING & BODY MASS INDEX (BMI) CALCULATION")
    print("=" * 70)

    # When data comes from CSVs, web APIs, or user input, numbers are often strings!
    raw_weight_kg = "78.5"
    raw_height_m = "1.75"

    # Converting strings to floating-point numbers (Type Casting)
    weight_kg = float(raw_weight_kg)
    height_m = float(raw_height_m)

    # BMI Formula: weight (kg) / (height (m) ** 2)
    bmi = weight_kg / (height_m ** 2)
    print(f"Weight: {weight_kg} kg | Height: {height_m} m")
    print(f"Calculated BMI: {bmi:.2f}")

    # Converting to integer (truncates decimal portion)
    bmi_rounded = int(bmi)
    print(f"Integer Truncated BMI: {bmi_rounded}")


# ==============================================================================
# SECTION 3: OPERATORS & LOGIC (FOOD DELIVERY SURGE ENGINE)
# ==============================================================================
def demo_section_3():
    print("\n" + "=" * 70)
    print("SECTION 3: ARITHMETIC, COMPARISON & LOGICAL OPERATORS")
    print("=" * 70)

    base_fee = 3.99
    distance_miles = 6.4
    per_mile_rate = 1.25
    is_rush_hour = True
    is_bad_weather = False
    customer_is_vip = True

    # Arithmetic operators
    distance_charge = distance_miles * per_mile_rate
    subtotal = base_fee + distance_charge

    # Floor division and modulo demonstration
    total_minutes = 74
    hours = total_minutes // 60  # Integer division: 1
    minutes = total_minutes % 60  # Remainder: 14
    print(f"Estimated Delivery Time: {hours} hr {minutes} mins ({total_minutes} total mins)")

    # Logical operators: and, or, not
    surge_multiplier = 1.0
    if is_rush_hour or is_bad_weather:
        surge_multiplier = 1.5

    # VIP discount applies only if not already under extreme surge
    if customer_is_vip and (surge_multiplier <= 1.5):
        discount = 2.00
    else:
        discount = 0.00

    final_delivery_fee = (subtotal * surge_multiplier) - discount
    print(f"Base Fee:        ${base_fee:.2f}")
    print(f"Distance Charge: ${distance_charge:.2f} ({distance_miles} miles @ ${per_mile_rate}/mi)")
    print(f"Surge Factor:    {surge_multiplier}x (Rush Hour: {is_rush_hour}, Storm: {is_bad_weather})")
    print(f"VIP Discount:   -${discount:.2f}")
    print(f"Final Total:     ${final_delivery_fee:.2f}")


# ==============================================================================
# SECTION 4: CONDITIONALS (PATIENT TRIAGE & SEVERITY CLASSIFICATION)
# ==============================================================================
def classify_patient_triage(heart_rate: int, oxygen_saturation: float) -> str:
    """
    Classifies a patient into clinical priority levels using if-elif-else logic.
    """
    if oxygen_saturation < 90.0 or heart_rate > 130:
        return "RED: Critical / Immediate Resuscitation"
    elif (90.0 <= oxygen_saturation < 94.0) or (110 < heart_rate <= 130):
        return "YELLOW: Urgent / Expedited Physician Review"
    elif oxygen_saturation >= 94.0 and (60 <= heart_rate <= 100):
        return "GREEN: Normal / Standard Care"
    else:
        return "BLUE: Low Priority / Ambulatory Observation"

def demo_section_4():
    print("\n" + "=" * 70)
    print("SECTION 4: DECISION MAKING WITH IF-ELIF-ELSE (TRIAGE CLASSIFIER)")
    print("=" * 70)

    test_patients = [
        {"id": "P-01", "hr": 145, "spo2": 88.5},
        {"id": "P-02", "hr": 115, "spo2": 93.0},
        {"id": "P-03", "hr": 72,  "spo2": 98.0},
        {"id": "P-04", "hr": 55,  "spo2": 96.0},
    ]

    for p in test_patients:
        triage_status = classify_patient_triage(p["hr"], p["spo2"])
        print(f"Patient {p['id']} (HR: {p['hr']} bpm, SpO2: {p['spo2']}%) -> {triage_status}")


# ==============================================================================
# SECTION 5: WHILE LOOPS (EV BATTERY DRAIN & RAPID CHARGING SIMULATION)
# ==============================================================================
def demo_section_5():
    print("\n" + "=" * 70)
    print("SECTION 5: WHILE LOOPS WITH BREAK & CONTINUE (EV FLEET CHARGING)")
    print("=" * 70)

    battery_level = 18.0     # Starting battery percentage
    target_charge = 80.0     # Rapid charging cutoff threshold
    minute = 0

    print(f"Initial Vehicle Battery: {battery_level:.1f}% | Target: {target_charge:.1f}%")

    while battery_level < target_charge:
        minute += 5
        # Fast charging rates: 12% per 5-min block below 50%, tapering to 6% above 50%
        if battery_level < 50.0:
            charge_added = 11.5
        else:
            charge_added = 6.0

        battery_level += charge_added
        print(f"  Minute {minute:02d}: Charged +{charge_added:.1f}% -> Current Battery: {min(battery_level, 100.0):.1f}%")

        # Safety override: Emergency thermal cutoff
        if minute >= 45:
            print("  [ALERT] 45-minute safety threshold reached. Disconnecting charger.")
            break

    print(f">> Charging Session Complete! Final Level: {min(battery_level, 100.0):.1f}% in {minute} minutes.")


# ==============================================================================
# SECTION 6: FOR LOOPS, ENUMERATE() & ZIP() (RETAIL SALES AUDITING)
# ==============================================================================
def demo_section_6():
    print("\n" + "=" * 70)
    print("SECTION 6: FOR LOOPS, RANGE, ENUMERATE & ZIP (SALES AUDIT)")
    print("=" * 70)

    stores = ["Downtown Metro", "Suburban Mall", "Airport Terminal", "Tech Center Hub"]
    daily_sales = [12450.00, 8920.50, 15300.75, 11840.25]
    sales_targets = [10000.00, 9500.00, 14000.00, 11000.00]

    print("--- 1. Using zip() to iterate through multiple sequences in parallel ---")
    total_sales = 0.0
    for store, actual, target in zip(stores, daily_sales, sales_targets):
        diff = actual - target
        total_sales += actual
        status_flag = "[EXCEEDED TARGET]" if diff >= 0 else "[MISSED TARGET]"
        print(f"Store: {store:<18} Actual: ${actual:>9,.2f} | Target: ${target:>9,.2f} ({diff:>+8,.2f}) {status_flag}")

    print(f"\nTotal Network Revenue: ${total_sales:,.2f}")

    print("\n--- 2. Using enumerate() to track iteration index & rank stores ---")
    sorted_pairs = sorted(zip(daily_sales, stores), reverse=True)
    for rank, (sales, name) in enumerate(sorted_pairs, start=1):
        print(f"Rank #{rank}: {name:<18} - ${sales:,.2f}")


# ==============================================================================
# SECTION 7: MODULAR FUNCTIONS (FINTECH LOAN DEBT-TO-INCOME EVALUATOR)
# ==============================================================================
def evaluate_loan_application(applicant_name: str, monthly_income: float, monthly_debt: float, credit_score: int, max_dti: float = 0.43) -> tuple:
    """
    Evaluates loan qualification based on Debt-to-Income (DTI) ratio and credit score.
    Returns: (is_approved: bool, dti_ratio: float, reason_code: str)
    """
    if monthly_income <= 0:
        return False, 0.0, "INVALID_INCOME"

    dti = monthly_debt / monthly_income

    if credit_score < 600:
        return False, dti, "SUBPRIME_CREDIT_SCORE"
    elif dti > max_dti:
        return False, dti, "EXCESSIVE_DTI_RATIO"
    else:
        return True, dti, "APPROVED_STANDARD_TERMS"

def demo_section_7():
    print("\n" + "=" * 70)
    print("SECTION 7: REUSABLE FUNCTIONS & RETURNING MULTIPLE VALUES")
    print("=" * 70)

    applicants = [
        {"name": "Sarah Chen",  "income": 7500.0, "debt": 1850.0, "score": 740},
        {"name": "Marcus Vance", "income": 5200.0, "debt": 2750.0, "score": 680},
        {"name": "Elena Rostova","income": 8900.0, "debt": 2100.0, "score": 580},
    ]

    for app in applicants:
        approved, dti, reason = evaluate_loan_application(
            app["name"], app["income"], app["debt"], app["score"]
        )
        decision = "APPROVED" if approved else "DENIED"
        print(f"Applicant: {app['name']:<15} Income: ${app['income']:<6} Debt: ${app['debt']:<6} Credit: {app['score']}")
        print(f"  -> Decision: {decision:<8} | DTI: {dti*100:.1f}% | Reason: {reason}\n")


# ==============================================================================
# 🎯 YOUR TURN TO CODE! HANDS-ON MICRO-EXERCISES (WITH SOLUTIONS)
# ==============================================================================
def exercise_1_solution():
    """
    Exercise 1: Smart Thermostat Comfort Index
    Write logic to calculate average daily temperature and classify heating/cooling mode.
    """
    morning_temp = 62.5
    afternoon_temp = 78.0
    evening_temp = 68.5

    # 1. Compute arithmetic average
    avg_temp = (morning_temp + afternoon_temp + evening_temp) / 3

    # 2. Determine mode
    if avg_temp < 65.0:
        mode = "HEATING REQUIRED"
    elif avg_temp > 75.0:
        mode = "AC COOLING REQUIRED"
    else:
        mode = "ECO / VENTILATION"

    print("--- Exercise 1 Solution: Smart Thermostat ---")
    print(f"Average Daily Temperature: {avg_temp:.1f}°F -> System Mode: {mode}")

def exercise_2_solution():
    """
    Exercise 2: Automated Customer Loyalty Tier Upgrade
    Loop through customer annual purchases and compute loyalty reward points.
    Rules:
      - Spend >= $5,000 -> Platinum (3 points per dollar)
      - Spend >= $2,000 -> Gold (2 points per dollar)
      - Spend <  $2,000 -> Silver (1 point per dollar)
    """
    customers = ["Acme Corp", "TechNova", "GlobalLogistics", "BioHealth", "ApexRetail"]
    annual_spend = [6450.0, 1850.0, 12200.0, 4300.0, 950.0]

    print("\n--- Exercise 2 Solution: Loyalty Points Ledger ---")
    for name, spend in zip(customers, annual_spend):
        if spend >= 5000.0:
            tier = "Platinum"
            points = int(spend * 3)
        elif spend >= 2000.0:
            tier = "Gold"
            points = int(spend * 2)
        else:
            tier = "Silver"
            points = int(spend * 1)
        print(f"Customer: {name:<16} Spend: ${spend:>9,.2f} | Tier: {tier:<8} | Points: {points:>6,}")


if __name__ == "__main__":
    demo_section_1()
    demo_section_2()
    demo_section_3()
    demo_section_4()
    demo_section_5()
    demo_section_6()
    demo_section_7()
    exercise_1_solution()
    exercise_2_solution()
