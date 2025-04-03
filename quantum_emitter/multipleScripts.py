import subprocess


def execute_scripts(script_paths):
    """
    Executes a list of Python scripts.
    :param script_paths: List of paths to Python scripts.
    """
    if not script_paths:
        print("No scripts provided.")
        return

    for script_path in script_paths:
        print(f"Executing {script_path}...")
        try:
            result = subprocess.run(["python", script_path], capture_output=True, text=True)
            print(f"Output of {script_path}:")
            print(result.stdout)
            if result.returncode != 0:
                print(f"Error in {script_path}:")
                print(result.stderr)
            print("-" * 40)
        except Exception as e:
            print(f"An error occurred while executing {script_path}: {e}")
            print("-" * 40)


if __name__ == "__main__":
    # Example list of script paths
    list_of_scripts = [
        "./kernelConvergenceECM.py",
        "./kernelConvergenceLifetime.py",
        "./spectralDensityConvergence.py"
    ]

    execute_scripts(list_of_scripts)