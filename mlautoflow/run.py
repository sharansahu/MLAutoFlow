import subprocess
import platform
import os
import sys

from create_cog_yaml import create_cog_components
from gpt_repo_loader import *

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

def process_git_repo(repo_path, preamble_file, output_file_path):
    ignore_file_path = os.path.join(repo_path, ".gptignore")

    if not os.path.exists(ignore_file_path):
        HERE = os.path.dirname(os.path.abspath(__file__))
        ignore_file_path = os.path.join(HERE, ".gptignore")

    if os.path.exists(ignore_file_path):
        ignore_list = get_ignore_list(ignore_file_path)
    else:
        ignore_list = []

    with open(output_file_path, 'w') as output_file:
        if preamble_file:
            with open(preamble_file, 'r') as pf:
                preamble_text = pf.read()
                output_file.write(f"{preamble_text}\n")
        process_repository(repo_path, ignore_list, output_file)
        output_file.write("--END--")

if __name__ == "__main__":
    install_cog()
    was_initialized = init_cog()
    if not was_initialized:
        repo_path = sys.argv[1] if len(sys.argv) >= 2 else os.getcwd()
        preamble_file = sys.argv[sys.argv.index("-p") + 1] if "-p" in sys.argv else None
        output_file_path = sys.argv[sys.argv.index("-o") + 1] if "-o" in sys.argv else 'output.txt'

        process_git_repo(repo_path, preamble_file, output_file_path)

        create_cog_components()
        print("Installation Done \U0001F604")
