import subprocess
import platform
import os

from create_cog_yaml import create_cog_components

CWD = os.getcwd()

def is_cog_installed():
    try:
        subprocess.run(["cog", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception:
        return False

def install_cog():
    if is_cog_installed():
        return

    os_type = platform.system()
    arch = platform.machine()
    download_url = ""

    if os_type == "Linux" or os_type == "Darwin":
        download_url = f"https://github.com/replicate/cog/releases/latest/download/cog_{os_type}_{arch}"
        try:
            subprocess.run(f"curl -o /usr/local/bin/cog -L {download_url}", shell=True, check=True)
            subprocess.run("chmod +x /usr/local/bin/cog", shell=True, check=True)
            print("Cog installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing Cog: {e}")

    elif os_type == "Windows":
        print("Unfortunately, Cog is not supported for Windows \U0001F622")

def init_cog():
    was_initialized = True
    if is_cog_installed():
        try:
            subprocess.run(["cog", "init"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            was_initialized = False
        except subprocess.CalledProcessError as e:
            print(f"Cog already initialized")
    return was_initialized

if __name__ == "__main__":
    install_cog()
    was_initialized = init_cog()
    create_cog_components()
    if not was_initialized:
        print("Installation Done \U0001F604")

        
