import copy
from time import time

import numpy as np
import torch
from equistore import Labels, TensorBlock, TensorMap
from rascaline import SoapPowerSpectrum
from LE_ps import get_LE_ps


def _block_to_torch(block, structure_i):
    assert block.samples.names[0] == "structure"
    samples = (
        block.samples.view(dtype=np.int32).reshape(-1, len(block.samples.names)).copy()
    )
    samples[:, 0] = structure_i
    samples = Labels(block.samples.names, samples)

    new_block = TensorBlock(
        values=torch.tensor(block.values).to(dtype=torch.get_default_dtype()),
        samples=samples,
        components=block.components,
        properties=block.properties,
    )

    for parameter in block.gradients_list():
        gradient = block.gradient(parameter)

        gradient_samples = (
            gradient.samples.view(dtype=np.int32)
            .reshape(-1, len(gradient.samples.names))
            .copy()
        )
        if gradient.samples.names == ("sample", "structure", "atom"):
            gradient_samples[:, 1] = structure_i
        gradient_samples = Labels(gradient.samples.names, gradient_samples)

        new_block.add_gradient(
            parameter=parameter,
            data=torch.tensor(gradient.data).to(dtype=torch.get_default_dtype()),
            samples=gradient_samples,
            components=gradient.components,
        )

    return new_block


def _move_to_torch(tensor_map, structure_i):
    blocks = []
    for _, block in tensor_map:
        blocks.append(_block_to_torch(block, structure_i))

    return TensorMap(tensor_map.keys, blocks)


class AtomisticDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        frames,
        all_species,
        spline_file, 
        E_nl, 
        E_max_2, 
        rcut,
        energies,
    ):
        all_center_species = Labels(
            names=["species_center"],
            values=np.array(all_species, dtype=np.int32).reshape(-1, 1),
        )
        all_neighbor_species_1 = Labels(
            names=["species_neighbor_1"],
            values=np.array(all_species, dtype=np.int32).reshape(-1, 1),
        )
        all_neighbor_species_2 = Labels(
            names=["species_neighbor_2"],
            values=np.array(all_species, dtype=np.int32).reshape(-1, 1),
        )

        self.ps = []
        self.index_map = []

        for frame_i, frame in enumerate(frames):
            if frame_i%100 == 0: print(frame_i)
            ps_i = get_LE_ps(frame, spline_file, E_nl, E_max_2, rcut)
            #ps_i.keys_to_properties(all_neighbor_species_1)
            #ps_i.keys_to_properties(all_neighbor_species_2)
            self.ps.append(
                _move_to_torch(ps_i, frame_i)
            )
            self.index_map.append(frame_i)

        energies = energies.reshape((energies.shape[0], 1))
        assert isinstance(energies, torch.Tensor)
        assert energies.shape == (len(frames), 1)
        self.energies = energies

    def __len__(self):
        return len(self.ps)

    def __getitem__(self, idx):
        data = (
            self.ps[idx],
            self.energies[idx],
            self.index_map[idx],
        )
        return data


def _collate_tensor_map(tensors, device):
    key_names = tensors[0].keys.names
    sample_names = tensors[0].block(0).samples.names
    if tensors[0].block(0).has_gradient("positions"):
        grad_sample_names = tensors[0].block(0).gradient("positions").samples.names
    unique_keys = set()
    for tensor in tensors:
        unique_keys.update(set(tensor.keys.tolist()))
    unique_keys = [tuple(k) for k in unique_keys]
    unique_keys.sort()
    values_dict = {key: [] for key in unique_keys}
    samples_dict = {key: [] for key in unique_keys}
    properties_dict = {key: None for key in unique_keys}
    components_dict = {key: None for key in unique_keys}
    grad_values_dict = {key: [] for key in unique_keys}
    grad_samples_dict = {key: [] for key in unique_keys}
    grad_components_dict = {key: None for key in unique_keys}
    previous_samples_count = {key: 0 for key in unique_keys}

    for tensor in tensors:
        for key, block in tensor:
            key = tuple(key)
            if components_dict[key] is None:
                # components and properties must be the same for each block of
                # the same key.
                components_dict[key] = block.components
                properties_dict[key] = block.properties
            values_dict[key].append(block.values)

            samples = np.asarray(block.samples.tolist())
            samples_dict[key].append(samples)

            if block.has_gradient("positions"):
                gradient = block.gradient("positions")
                if grad_components_dict[key] is None:
                    grad_components_dict[key] = gradient.components
                grad_values_dict[key].append(gradient.data)

                grad_samples = np.asarray(gradient.samples.tolist())
                grad_samples[:, 0] += previous_samples_count[key]
                grad_samples_dict[key].append(grad_samples)

            previous_samples_count[key] += samples.shape[0]

    blocks = []
    for key in unique_keys:
        block = TensorBlock(
            values=torch.vstack(values_dict[key]).to(device),
            samples=Labels(
                names=sample_names,
                values=np.asarray(np.vstack(samples_dict[key]), dtype=np.int32),
            ),
            components=components_dict[key],
            properties=properties_dict[key],
        )
        if grad_components_dict[key] is not None:
            block.add_gradient(
                "positions",
                data=torch.vstack(grad_values_dict[key]).to(device),
                components=grad_components_dict[key],
                samples=Labels(
                    names=grad_sample_names,
                    values=np.asarray(
                        np.vstack(grad_samples_dict[key]), dtype=np.int32
                    ),
                ),
            )
        blocks.append(block)

    return TensorMap(Labels(key_names, np.asarray(unique_keys, dtype=np.int32)), blocks)


def _collate_data(device, dataset):

    def do_collate(data):
        ps = _collate_tensor_map([d[0] for d in data], device)
        energies = torch.vstack([d[1] for d in data]).to(device=device)
        indices = np.array([d[2] for d in data])
        return ps, energies, indices

    return do_collate


def create_dataloader(dataset, batch_size, shuffle=True, device="cpu"):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_data(device, dataset),
    )