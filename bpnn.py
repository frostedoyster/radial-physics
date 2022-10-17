import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from rascaline import SoapPowerSpectrum
from equistore import Labels, TensorBlock, TensorMap

from dataset_processing import get_dataset_slices
from error_measures import get_sse, get_rmse, get_mae
from dataset_bpnn import AtomisticDataset, create_dataloader

torch.set_num_threads(16)
print("CUDA is available: ", torch.cuda.is_available())  # See if we can use a GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# torch.set_default_dtype(torch.float64)
# torch.manual_seed(1234)
RANDOM_SEED = 1000
np.random.seed(RANDOM_SEED)
print(f"Random seed: {RANDOM_SEED}", flush = True)

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCALMOL = 627.5
EV_TO_KCALMOL = HARTREE_TO_KCALMOL/HARTREE_TO_EV

DATASET_PATH = 'datasets/random-ch4-10k.extxyz'
TARGET_KEY = "energy" # "elec. Free Energy [eV]" # "U0"
CONVERSION_FACTOR = HARTREE_TO_KCALMOL

n_test = 1000
n_train = 1000

n_validation_splits = 10
assert n_train % n_validation_splits == 0
n_validation = n_train // n_validation_splits
n_train_sub = n_train - n_validation

test_slice = str(0) + ":" + str(n_test)
train_slice = str(n_test) + ":" + str(n_test+n_train)

# Spherical expansion and composition

def get_composition_features(frames, all_species):
    species_dict = {s: i for i, s in enumerate(all_species)}
    data = torch.zeros((len(frames), len(species_dict)))
    for i, f in enumerate(frames):
        for s in f.numbers:
            data[i, species_dict[s]] += 1
    properties = Labels(
        names=["atomic_number"],
        values=np.array(list(species_dict.keys()), dtype=np.int32).reshape(
            -1, 1
        ),
    )

    frames_i = np.arange(len(frames), dtype=np.int32).reshape(-1, 1)
    samples = Labels(names=["structure"], values=frames_i)

    block = TensorBlock(
        values=data, samples=samples, components=[], properties=properties
    )
    composition = TensorMap(Labels.single(), blocks=[block])
    return composition.block().values

'''
hypers_rs = {
    "cutoff": 4.5,
    "max_radial": 12,
    "atomic_gaussian_width": 0.05,
    "center_atom_weight": 0.0,
    "radial_basis": {"Gto": {"spline_accuracy": 1e-8}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "radial_scaling":  {"Willatt2018": { "scale": 2.0, "rate": 2.0, "exponent": 6}},
}
'''

hypers_ps = {
    "cutoff": 6.0,
    "max_radial": 8,
    "max_angular": 6,
    "atomic_gaussian_width": 0.2,
    "center_atom_weight": 0.0,
    "radial_basis": {"Gto": {"spline_accuracy": 1e-8}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "radial_scaling":  {"Willatt2018": { "scale": 2.0, "rate": 2.0, "exponent": 6}},
}

calculator_ps = SoapPowerSpectrum(**hypers_ps)
# calculator_rs = SoapRadialSpectrum(**hypers_rs)

train_structures, test_structures = get_dataset_slices(DATASET_PATH, train_slice, test_slice)

def move_to_torch(rust_map: TensorMap) -> TensorMap:
    torch_blocks = []
    for _, block in rust_map:
        torch_block = TensorBlock(
            values=torch.tensor(block.values).to(dtype=torch.get_default_dtype()),
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )
        torch_blocks.append(torch_block)
    return TensorMap(
            keys = rust_map.keys,
            blocks = torch_blocks
            )

print("Calculating invariants", flush = True)

# train_rs = calculator_rs.compute(train_structures)
# train_rs = move_to_torch(train_rs)

# test_rs = calculator_rs.compute(test_structures)
# test_rs = move_to_torch(test_rs)

train_ps = calculator_ps.compute(train_structures)
train_ps = move_to_torch(train_ps)

test_ps = calculator_ps.compute(test_structures)
test_ps = move_to_torch(test_ps)

all_species = np.unique(np.concatenate([train_ps.keys["species_center"], test_ps.keys["species_center"]]))

print("Calculating composition features", flush = True)
train_comp = get_composition_features(train_structures, all_species)
test_comp = get_composition_features(test_structures, all_species)
print("Composition features done", flush = True)


train_energies = [structure.info[TARGET_KEY] for structure in train_structures]
train_energies = torch.tensor(train_energies, dtype = torch.get_default_dtype()) * CONVERSION_FACTOR

test_energies = [structure.info[TARGET_KEY] for structure in test_structures]
test_energies = torch.tensor(test_energies, dtype = torch.get_default_dtype()) * CONVERSION_FACTOR

print("Calculating composition features", flush = True)
X_train = get_composition_features(train_structures, all_species)
X_test = get_composition_features(test_structures, all_species)
print("Composition features done", flush = True)

# nu = 0 contribution
if "methane" in DATASET_PATH or "ch4" in DATASET_PATH:
    mean_train_energy = torch.mean(train_energies)
    train_energies -= mean_train_energy
    test_energies -= mean_train_energy
else:
    c_comp = torch.linalg.solve(X_train.T @ X_train, X_train.T @ train_energies)
    train_energies -= X_train @ c_comp
    test_energies -= X_test @ c_comp

# Need normalization!!!!!!!!!
  
print("Creating datasets and dataloaders")  
train_dataset = AtomisticDataset(
    train_structures,
    all_species,
    hypers=hypers_ps,
    energies=train_energies,
)

test_dataset = AtomisticDataset(
    test_structures,
    all_species,
    hypers=hypers_ps,
    energies=test_energies,
)

train_dataloader = create_dataloader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    device=device,
)

test_dataloader = create_dataloader(
    test_dataset,
    batch_size=16,
    shuffle=True,
    device=device,
)

for ps, energies, indices in train_dataloader:
    nfeat = ps.block(0).values.shape[-1]
print(f"nfeat: {nfeat}")

nrepeat = 10
avgerr = 0.0
nlayers = 3
nneurons = [32, 32, 32]
assert nlayers == len(nneurons)

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.NNs = nn.ModuleDict()
        for species in all_species:  # For each element, create a NN.
            self.NNs[str(species)] = nn.ModuleList()  # Initialise a list that will contain all the layers.
            # NNs[str(species)].append(nn.Linear(nfeat, 1, dtype = torch.float32, bias = False))
            self.NNs[str(species)].append(nn.Linear(nfeat, nneurons[0], dtype = torch.float32))  # Create transformation between input layer and first hidden layer.
            for j in range(nlayers-1):
                self.NNs[str(species)].append(nn.Linear(nneurons[j], nneurons[j+1], dtype = torch.float32))  # Create transformations between hidden layers..
            self.NNs[str(species)].append(nn.Linear(nneurons[nlayers-1], 1, dtype = torch.float32))  # Create tranformation between last hidden layer and output layer.

    def forward(self, tensor_map, original_indices):

        y_structure = torch.zeros(size = (len(original_indices), 1), device = tensor_map.block(0).values.device)

        for species in all_species:
            NN_for_current_center_species = self.NNs[str(species)]
            try:
                block = tensor_map.block(species_center=species)
            except ValueError as e:
                # print(f"Batch doesn't have element {species}")
                continue
            x = block.values

            # ylin = self.NNs[str(species)][0](x)
            yprov = x
            for k in range(nlayers):
                yprov = torch.tanh(NN_for_current_center_species[k](yprov))  # +1
            yprov = NN_for_current_center_species[nlayers](yprov)  # +1
            y = yprov  # + ylin

            indices_for_index_add = []
            for structure in block.samples["structure"]:
                indices_for_index_add.append(np.where(original_indices == structure)[0][0])
            indices_for_index_add = torch.tensor(indices_for_index_add, dtype=torch.int32)

            y_structure.index_add_(0, indices_for_index_add, y)

        return y_structure




for irepeat in range(nrepeat):       

    network = Network().to(device)
    optimizer = optim.Adam(network.parameters(), lr = 1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold = 1e-5, eps = 1e-12, patience = 50, verbose = True)


    errmin = 10000000.0
    best_params = {}
    for epoch in range(10000):
        initial_time = time.time()

        for ps, energies, original_indices in train_dataloader:
            optimizer.zero_grad()  # Avoid accumulation of gradients

            predictions = network(ps, original_indices)
            # print(predictions.requires_grad)
            loss = F.mse_loss(predictions, energies, reduction='sum')
            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            with torch.no_grad():
                train_loss = 0.0
                for ps, energies, original_indices in train_dataloader:
                    predictions = network(ps, original_indices)
                    # print(predictions.requires_grad)
                    train_loss += F.mse_loss(predictions, energies, reduction='sum').item()
                train_loss = np.sqrt(train_loss/n_train)
            with torch.no_grad():
                test_loss = 0.0
                for ps, energies, original_indices in test_dataloader:
                    predictions = network(ps, original_indices)
                    # print(predictions.requires_grad)
                    test_loss += F.mse_loss(predictions, energies, reduction='sum').item()
                test_loss = np.sqrt(test_loss/n_test)

        if (test_loss <= errmin):
            errmin = test_loss
            for name, params in network.named_parameters():
                best_params[name] = params.clone()
        
        lr = optimizer.param_groups[0]["lr"]  # This appears to be making a deep copy, for whatever reason (probably copying from GPU to CPU).
        # np.set_printoptions(precision=3)  # If this is not set the default precision will be 4 and torch.float64 numbers will look like float32 numbers.
        print(repr((epoch+1)).rjust(6), repr(train_loss).rjust(20), repr(test_loss).rjust(20), lr, time.time()-initial_time, flush = "True")

        scheduler.step(test_loss)
        if (optimizer.param_groups[0]["lr"] <= 2e-12):
            print("Very small learning rate reached: 1e-12")
            break
        if (optimizer.param_groups[0]["lr"] != lr):   
            for name, params in network.named_parameters():
                params.data.copy_(best_params[name])
            optimizer = optim.Adam(network.parameters(), lr = lr*0.1)  # Reinitialise Adam so that it resets the moment vectors.
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold = 1e-5, eps = 1e-12, patience = 50, verbose = True)  # Also reinitialise scheduler just in case.

    print(errmin)
    avgerr += errmin
avgerr = avgerr/nrepeat
with open("carbon-nn-LE-f1f2-large.dat", "a") as out:
    out.write(str(n_train) + " " + str(nfeat) + " " + str(avgerr) + "\n")

