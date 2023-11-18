import subprocess
import platform
import os 
import re

def is_ubuntu():
    os_info = platform.system().lower()
    if 'darwin' in os_info:
        return False  
    elif 'linux' in os_info:
        try:
            with open('/etc/os-release') as file:
                for line in file:
                    if line.startswith('NAME=') and 'ubuntu' in line.lower():
                        return True
                return False
        except FileNotFoundError:
            return False
    else:
        return False


def update_apt_cache():
    try:
        subprocess.run(['sudo', 'apt-get', 'update'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while updating apt cache: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def get_system_packages():
    package_list = []
    apt_lists_dir = '/var/lib/apt/lists/'

    try:
        for filename in os.listdir(apt_lists_dir):
            if filename.endswith('_Packages'):
                package_file_path = os.path.join(apt_lists_dir, filename)
                with open(package_file_path) as file:
                    package_list.extend(re.findall(r"^Package: (.+)$", file.read(), re.MULTILINE))
        package_list = sorted(set(package_list))
        return package_list
    except Exception as e:
        print(f"Error reading package lists: {e}")
        return []
        