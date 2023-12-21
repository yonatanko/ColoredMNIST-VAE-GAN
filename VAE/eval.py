from hw2_212984801_train import *
import io

transform = transforms.Compose([transforms.ToTensor()])
colored_mnist_train = ColoredMNIST(root='./data', train=True, download=True, transform=transform)
colored_mnist_test = ColoredMNIST(root='./data', train=False, download=True)
test_loader = torch.utils.data.DataLoader(dataset=colored_mnist_test, batch_size=120, shuffle=False, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_continuous(model):
    plot_digits_continuous(model)


def evaluate_discrete(model):
    plot_digits_discrete(model)


def evaluate_joint(model):
    plot_digits_joint(model)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=lambda storage, loc: storage)
        else:
            return super().find_class(module, name)

def main():
    #load continuous model
    if device == "coda:0":
        model = pickle.load(open('vae_continuous.pkl', 'rb'))
    else:
        model = CPU_Unpickler(open('vae_continuous.pkl', 'rb')).load()

    model.to(device)
    model.eval()
    evaluate_continuous(model)

    #load discrete model
    if device == "coda:0":
        model = pickle.load(open('discrete_vae.pkl', 'rb'))
    else:
        model = CPU_Unpickler(open('discrete_vae.pkl', 'rb')).load()

    model.to(device)
    model.eval()
    evaluate_discrete(model)

    #load joint model
    if device == "coda:0":
        model = pickle.load(open('joint_vae.pkl', 'rb'))
    else:
        model = CPU_Unpickler(open('joint_vae.pkl', 'rb')).load()

    model.to(device)
    model.eval()
    evaluate_joint(model)

if __name__ == '__main__':
    main()
