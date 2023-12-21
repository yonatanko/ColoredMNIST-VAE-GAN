from hw2_212984801_q2_train import *
import io

transform = transforms.Compose([transforms.ToTensor()])
colored_mnistA_train = ColoredMNISTA(root='./data', train=True, download=True, transform=transform)
colored_mnistA_test = ColoredMNISTA(root='./data', train=False, download=True, transform = transform)
test_loader_A = torch.utils.data.DataLoader(dataset=colored_mnistA_test, batch_size=120, shuffle=False, num_workers = 4)

colored_mnistB_train = ColoredMNISTB(root='./data', train=True, download=True, transform=transform)
colored_mnistB_test = ColoredMNISTB(root='./data', train=False, download=True, transform = transform)
test_loader_B = torch.utils.data.DataLoader(dataset=colored_mnistB_test, batch_size=120, shuffle=False, num_workers = 4, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=lambda storage, loc: storage)
        else:
            return super().find_class(module, name)


def evaluate_cycle_gan():
    # load all generators
    if device == "cuda:0":
        ab_gen = pickle.load(open('ab_gen.pkl', 'rb'))
        ba_gen = pickle.load(open('ba_gen.pkl', 'rb'))
    else:
        ab_gen = CPU_Unpickler(open('ab_gen.pkl', 'rb')).load()
        ba_gen = CPU_Unpickler(open('ba_gen.pkl', 'rb')).load()

    ab_gen.to(device)
    ba_gen.to(device)
    ab_gen.eval()
    ba_gen.eval()

    # take 2 photos from each dataset and use generators to create new photos
    i = 0
    for a,b in zip(test_loader_A, test_loader_B):
        a = a[0].to(device)
        b = b[0].to(device)
        generate_imgs(a, b, ab_gen, ba_gen)
        i+=1
        if i == 3:
            break


if __name__ == '__main__':
    evaluate_cycle_gan()