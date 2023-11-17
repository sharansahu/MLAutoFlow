# MLAutoFlow: Automatically Push Models To Replicate

MLAutoFlow is a package that allows users to push their custom-trained machine-learning models to Replicate without any installations or hassles. This tool is particularly useful for data scientists and developers who want to get their open-source machine learning models deployed fast without any hassles.

## Key Features
1) **Automatic Cog Installation:** Installs Cog if it's not already installed on the system.
2) **Automatic `cog.yaml` Generation:** Generates a `cog.yaml` file based on the user's environment, specifically targeting Python packages (from `requirements.txt` or via `pipreqs`) and system packages (for Ubuntu users).
3) **Simple Execution:** All setup steps are handled by running a single Python script.
4) **(Future) LLM Integration:** Future updates aim to use Large Language Models (LLMs) to analyze user code and automatically create the `predict.py` file, further simplifying the process of pushing models to Replicate.
5) **(Future) Deploying To Replicate:** In future updates, you'll be able to automatically push your model to Replicate

## Installation
To install MLAutoFlow, simply run:

```
pip install -r requirements.txt
```

## Usage
After installation, you can start using MLAutoFlow by running:

```
python run.py
```
This script performs the following actions:

1) Checks if Cog is installed and installs it if necessary.
2) Initialize Cog in the current directory if it hasn't been done already.
3) Generates a predict.py file as a placeholder for your model's prediction logic.
4) Creates a cog.yaml file tailored to your environment, including your Python version, detected Python packages, and System packages (if running on Ubuntu).

from your project directory
