import torch
import torchvision
import torchvision.transforms as transforms
from model_soup.soup.evaluator import Evaluator


class CIFAR100Evaluator(Evaluator):
    def __init__(self, device=None, batch_size=256, input_size=32):
        
        super().__init__(device=device)

        data_transforms = {
            'test': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        testset = torchvision.datasets.CIFAR100(root='./data/CIFAR100', train=False,
                                       download=True, transform=data_transforms["test"])

        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


    def eval_func(self, model):
        '''
        Eval function to test the model with the given weights over the test dataset
        '''
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total
