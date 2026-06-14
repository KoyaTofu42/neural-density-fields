import os
import shutil

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Create directories
    dirs_to_create = [
        "configs", "scripts", "src/data", "src/physics", "src/training", "src/utils"
    ]
    for d in dirs_to_create:
        os.makedirs(os.path.join(root, d), exist_ok=True)
        print(f"Created directory: {d}")

    # 2. Move files from src/dataset and src/data_gen to src/data
    src_data = os.path.join(root, "src", "data")
    for old_dir in ["src/dataset", "src/data_gen"]:
        full_old_dir = os.path.join(root, old_dir)
        if os.path.exists(full_old_dir):
            for file in os.listdir(full_old_dir):
                if file.endswith('.py'):
                    src_file = os.path.join(full_old_dir, file)
                    dst_file = os.path.join(src_data, file)
                    shutil.move(src_file, dst_file)
                    print(f"Moved {src_file} -> {dst_file}")
            # Remove old directory after moving files
            shutil.rmtree(full_old_dir)
            print(f"Deleted directory: {old_dir}")

    # 3. Move root scripts to scripts/
    scripts_to_move = [
        "train.py", "generate_data.py", "inspect_data.py", 
        "inspect_predictions.py", "overfit_batch.py", "main.py"
    ]
    scripts_dir = os.path.join(root, "scripts")
    for script in scripts_to_move:
        src_file = os.path.join(root, script)
        if os.path.exists(src_file):
            dst_file = os.path.join(scripts_dir, script)
            shutil.move(src_file, dst_file)
            print(f"Moved {script} -> scripts/{script}")
            
    print("Restructuring complete!")

if __name__ == "__main__":
    main()
