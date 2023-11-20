# Output predict.py file using LLM on example of training a simple MLP on MNIST data

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