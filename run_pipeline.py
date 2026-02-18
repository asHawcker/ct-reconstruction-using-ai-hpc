import os
import subprocess

# Paths
PHANTOM_DIR = "data/raw_phantoms"
INPUT_DIR = "data/dataset/input_iter10" 
TARGET_DIR = "data/dataset/target_iter50"
SOLVER_EXE = "./sirt_solver"

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(TARGET_DIR, exist_ok=True)

# 1. Generate Phantoms
print("--- Step 1: Generating Random Phantoms ---")
os.system("python3 src_ai/generate_dataset.py")

# 2. Compile C++
print("--- Step 2: Compiling HPC Solver ---")
ret = os.system(f"g++ -O3 -fopenmp src_hpc/sirt_solver.cpp -o {SOLVER_EXE}")
if ret != 0:
    print("Compilation failed!")
    exit()

# 3. Run Solver on all files
files = sorted([f for f in os.listdir(PHANTOM_DIR) if f.endswith('.bin')])
print(f"--- Step 3: Processing {len(files)} files with HPC Solver ---")

for f in files:
    input_path = os.path.join(PHANTOM_DIR, f)
    out10_path = os.path.join(INPUT_DIR, f)
    out50_path = os.path.join(TARGET_DIR, f)
    
    # Call C++: ./sirt_solver <in> <out10> <out50>
    cmd = [SOLVER_EXE, input_path, out10_path, out50_path]
    subprocess.run(cmd)

print("--- Data Generation Complete! ---")