import os
import yaml

from python_reqs import *
from ubuntu_reqs import *
from llm_module import extract_context, generate_predictor_code

CWD = os.getcwd()

def create_predict_file():
    output_path = os.path.join(CWD, "output.txt")
    context = extract_context(output_path)

    predictor_code = generate_predictor_code(context)
    with open('predict.py', 'w') as file:
        file.write(predictor_code.strip())

def format_packages(packages):
    return "\n".join(f'    - "{pkg}"' for pkg in packages)

def create_cog_components():
    create_predict_file()

    python_ver = python_version()
    python_pkgs = python_packages()
    system_pkgs = get_system_packages() if is_ubuntu() and update_apt_cache() else []

    if python_pkgs:
        python_pkgs_section = f"python_packages:\n{format_packages(python_pkgs)}"
    else:
        python_pkgs_section = (
            "# python_packages:\n"
            "  #   - \"numpy==1.19.4\"\n"
            "  #   - \"torch==1.8.0\"\n"
            "  #   - \"torchvision==0.9.0\""
        )

    if system_pkgs:
        system_pkgs_section = f"system_packages:\n{format_packages(system_pkgs)}"
    else:
        system_pkgs_section = (
            "# system_packages:\n"
            "  #   - \"libgl1-mesa-glx\"\n"
            "  #   - \"libglib2.0-0\""
        )

    cog_config_content = f"""# Configuration for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/yaml.md

build:
  # set to true if your model requires a GPU
  gpu: false

  {system_pkgs_section}

  # python version in the form '{python_ver}' or '{python_ver}.x'
  python_version: "{python_ver}"

  {python_pkgs_section}

  # commands run after the environment is setup
  # run:
  #   - "echo env is ready!"
  #   - "echo another command if needed"

# predict.py defines how predictions are run on your model
predict: "predict.py:Predictor"
"""

    with open(os.path.join(CWD, 'cog.yaml'), 'w') as file:
        file.write(cog_config_content)
    


    