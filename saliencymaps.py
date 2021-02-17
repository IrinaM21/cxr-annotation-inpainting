# flashtorch and torch 1.6 need to be installed
from flashtorch.saliency import Backprop
from flashtorch.utils import load_image, apply_transforms, denormalize, format_for_plotting

PATH = 'FILLER PATH'
input = apply_transforms(load_image(PATH))
# COVID-19 is 0, Normal is 1, Viral Pneumonia is 2
target_class = 0

MODEL = 'FILLER PATH'
model = torch.load(MODEL)

backprop = Backprop(model)

gradients = backprop.calculate_gradients(input, target_class)
max_gradients = backprop.calculate_gradients(input, target_class, take_max=True)
backprop.visualize(input, target_class)

guided_gradients = backprop.calculate_gradients(input, target_class, guided=True)
max_guided_gradients = backprop.calculate_gradients(input, target_class, take_max=True, guided=True)
# visualize
backprop.visualize(input, target_class)
