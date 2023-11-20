# MLAutoFlow: Automatically Push Models To Replicate

MLAutoFlow is a package that allows users to push their custom-trained machine-learning models to Replicate without any installations or hassles. This tool is particularly useful for data scientists and developers who want to get their open-source machine learning models deployed fast without any hassles.

## Key Features
1) **Automatic Cog Installation:** Installs Cog if it's not already installed on the system.
2) **Automatic `cog.yaml` Generation:** Generates a `cog.yaml` file based on the user's environment, specifically targeting Python packages (from `requirements.txt` or via `pipreqs`) and system packages (for Ubuntu users).
3) **Simple Execution:** All setup steps are handled by running a single Python script.
4) **LLM Integration:** Use Large Language Models (LLMs) to analyze user code and automatically create the `predict.py` file, further simplifying the process of pushing models to Replicate.

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
6) Extracts user code and uses OpenAI GPT-4 Turbo to automatically create the `predict.py` file.

from your project directory

## Example
The [`examples`](examples) folder has an example of training a simple MLP on MNIST data. After creating a `.env` file with your OpenAI API key, you can run

```
python ../run.py
```

in the example's folder. This is the sample `predict.py` file one may get out using this example

```
from cog import BasePredictor, Input, Path
from image_classification import LightningMNISTClassifier, DataLoader, transforms

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = LightningMNISTClassifier(lr_rate=1e-3)
        self.model.load_state_dict(torch.load("./trained-model/mnist_epoch=9-val_loss=-0.88.ckpt"))
        self.model.eval()

    def predict(
        self,
        image: Path = Input(description="Grayscale input image"),
        scale: float = Input(
            description="Factor to scale image by", ge=0, le=10, default=1.5
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        processed_input = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])(image)
        output = self.model(processed_input, scale)
        return output
```
As we can see, we get a reasonable predict.py file with the correct dependencies imported and only uses the functions provided within the codebase.

## Test Your Model Locally
To test that this works, try running a prediction on the model:

```
$ cog predict -i image=@input.jpg
✓ Building Docker image from cog.yaml... Successfully built 664ef88bc1f4
✓ Model running in Docker image 664ef88bc1f4

Written output to output.png
```

To pass more inputs to the model, you can add more -i options:

```
$ cog predict -i image=@image.jpg -i scale=2.0
```

In this case it’s just a number, not a file, so you don’t need the @ prefix.

## Push Your Model
Once you are done configuring and testing your model locally with the `cog.yaml` and `predict.py` files, you can create a corresponding model page on Replicate and publish it to Replicate’s registry:

```
cog login
cog push r8.im/<your-username>/<your-model-name>
```

Your username and model name must match the values you set on Replicate.

Note: You can also set the image property in your cog.yaml file. This allows you to run cog push without specifying the image, and also makes your model page on Replicate more discoverable for folks reading your model’s source code.

## Run Predictions
Once you’ve pushed your model to Replicate it will be visible on the website, and you can use the web-based form to run predictions using your model.

To run predictions in the cloud from your code, you can use the Python client library.

Install it from pip:

```
pip install replicate
```

Authenticate by setting your token in an environment variable:

```
export REPLICATE_API_TOKEN=<paste-your-token-here>
```

Find your API token in your account settings. Then, you can use the model from your Python code:

```
import replicate
replicate.run(
  "replicate/hello-world:5c7d5dc6dd8bf75c1acaa8565735e7986bc5b66206b55cca93cb72c9bf15ccaa",
  input={"text": "python"}
)

# "hello python"
```

To pass a file as an input, use a file handle or URL:

```
image = open("mystery.jpg", "rb")
# or...
image = "https://example.com/mystery.jpg"

replicate.run(
  "replicate/resnet:dd782a3d531b61af491d1026434392e8afb40bfb53b8af35f727e80661489767",
  input={"image": image}
)
```
URLs are more efficient if your file is already in the cloud somewhere, or it’s a large file.

