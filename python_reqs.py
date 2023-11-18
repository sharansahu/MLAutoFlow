import subprocess
import os 
import sys

CWD = os.getcwd()

def is_pipreqs_installed():
    try:
        subprocess.run(["pipreqs", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception:
        return False

def python_version():
    version_info = sys.version_info
    python_version = f'{version_info.major}.{version_info.minor}.{version_info.micro}'
    return python_version

def python_packages():
    requirements = []
    requirements_file = os.path.join(CWD, "requirements.txt")
    if os.path.exists(requirements_file):
        with open(requirements_file) as file:
            for line in file:
                requirements.append(line.strip())
    else:
        try:
            if not is_pipreqs_installed():
                subprocess.run(["pip", "install", "pipreqs"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            result = subprocess.run(['pipreqs', CWD, '--print'], stdout = subprocess.PIPE, stderr = subprocess.PIPE)
            requirements = result.stdout.decode('utf-8').splitlines()
        except subprocess.CalledProcessError as e:
            print(f"Error running pipreqs: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
    return requirements
