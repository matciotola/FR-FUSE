
import os
from scipy import io
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from network import FR_FUSE_Model
from losses import SpectralLoss, StructLoss
from input_preprocessing import normalize, denormalize, input_prepro_rr, input_prepro_fr

from Utils.dl_tools import open_config, generate_paths, TrainingDataset20m, get_patches


def fr_fuse(bands_10, bands_20):

    # Open config file
    config_path = 'config.yaml'

    config = open_config(config_path)

    # Define hyperparameters
    ratio = 2
    model_weights_path = config.model_weights_path

    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_number
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model definition
    net = FR_FUSE_Model(config.number_bands_10, config.number_bands_20)

    # Load model weights
    if not config.train or config.resume:
        if not model_weights_path:
            model_weights_path = os.path.join('weights', 'FR-FUSE.tar')
        net.load_state_dict(torch.load(model_weights_path))

    # Move model to device
    net = net.to(device)

    # Training step - if config.train is True
    if config.train:
        # Define the train Dataset and DataLoader
        train_paths_10, train_paths_20, _ = generate_paths(config.training_img_root, config.training_img_names)
        ds_train = TrainingDataset20m(train_paths_10, train_paths_20, normalize, input_prepro_rr, get_patches, ratio,
                                      config.training_patch_size_20, config.training_patch_size_10)
        train_loader = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True)

        # Define the validation Dataset and DataLoader - if config.validation_img_names is not empty
        if len(config.validation_img_names) != 0:
            val_paths_10, val_paths_20, _ = generate_paths(config.validation_img_root, config.validation_img_names)
            ds_val = TrainingDataset20m(val_paths_10, val_paths_20, normalize, input_prepro_rr, get_patches, ratio,
                                        config.training_patch_size_20, config.training_patch_size_10)
            val_loader = DataLoader(ds_val, batch_size=config.batch_size, shuffle=True)
        else:
            val_loader = None

        # Train the model
        net, history = train(device, net, train_loader, config, val_loader)

        # Save the model weights
        if config.save_weights:
            if not os.path.exists(config.save_weights_path):
                os.makedirs(config.save_weights_path)
            torch.save(net.state_dict(), config.save_weights_path)

        # Save the training stats
        if config.save_training_stats:
            if not os.path.exists('./Stats/FR-FUSE'):
                os.makedirs('./Stats/FR-FUSE')
            io.savemat('./Stats/FR-FUSE/Training_20m.mat', history)

    # Test step

    # Store mean and std of the test image
    mean = torch.mean(bands_20, dim=(2, 3), keepdim=True)
    std = torch.std(bands_20, dim=(2, 3), keepdim=True)

    # Normalize the test image
    bands_10_norm = normalize(bands_10)
    bands_20_norm = normalize(bands_20)

    # Preprocess the test image
    bands_10_norm, bands_20_norm, spec_ref, struct_ref = input_prepro_fr(bands_10_norm, bands_20_norm, ratio)

    # Move the test image to device
    input_10 = bands_10_norm.to(device)
    input_20 = bands_20_norm.to(device)
    spec_ref = spec_ref.to(device)
    struct_ref = struct_ref.to(device)

    # Target adaptation step - Mandatory as it is part of the algorithm
    net, ta_history = target_adaptation(device, net, input_10, input_20, spec_ref, struct_ref, config)

    # Save the model weights regarding the current test image - if config.ta_save_weights is True
    if config.ta_save_weights:
        if not os.path.exists(config.ta_save_weights_path):
            os.makedirs(config.ta_save_weights_path)
        torch.save(net.state_dict(), config.ta_save_weights_path)

    # Save the target adaptation stats regarding the current test image - if config.ta_save_training_stats is True
    if config.ta_save_training_stats:
        if not os.path.exists('./Stats/FR-FUSE'):
            os.makedirs('./Stats/FR-FUSE')
        io.savemat('./Stats/FR-FUSE/TA_R-Fuse.mat', ta_history)

    # Super-resolution step
    net.eval()
    with torch.no_grad():
        fused = net(input_10, input_20)

    # Denormalize the fused image
    fused = denormalize(fused, mean, std)

    # Clear the cache
    torch.cuda.empty_cache()

    # Return the fused image as a Tensor on CPU
    return fused.cpu().detach()


def train(device, net, train_loader, config, val_loader=None):

    # Define the loss function and the optimizer
    criterion = nn.L1Loss(reduction='mean')
    optim = torch.optim.Adam(net.parameters(), lr=config.learning_rate)

    # Move the model and loss to device
    net = net.to(device)
    criterion = criterion.to(device)

    # Initialize the history
    history_loss = []
    history_val_loss = []

    # Training loop
    pbar = tqdm(range(config.epochs))

    for epoch in pbar:

        pbar.set_description('Epoch %d/%d' % (epoch + 1, config.epochs))
        # Initialize the running loss
        running_loss = 0.0
        running_val_loss = 0.0

        # Set the model to train mode
        net.train()

        for i, data in enumerate(train_loader):

            # Zero the parameter gradients
            optim.zero_grad()

            # Move the data to device
            inputs_10, inputs_20, labels = data
            inputs_10 = inputs_10.to(device)
            inputs_20 = inputs_20.to(device)
            labels = labels.to(device)

            # Forward step
            outputs = net(inputs_10, inputs_20)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward step
            loss.backward()

            # Weights update
            optim.step()

            # Update the running loss
            running_loss += loss.item()

        # Compute the epoch loss
        running_loss = running_loss / len(train_loader)

        # Validation step - if val_loader is not None
        if val_loader is not None:
            # Set the model to eval mode
            net.eval()

            # Disable the gradient computation
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    # Move the data to device
                    inputs_10, inputs_20, labels = data
                    inputs_10 = inputs_10.to(device)
                    inputs_20 = inputs_20.to(device)
                    labels = labels.to(device)

                    # Forward step
                    outputs = net(inputs_10, inputs_20)

                    # Compute the loss
                    val_loss = criterion(outputs, labels)

                    # Update the running loss
                    running_val_loss += val_loss.item()

            # Compute the epoch validation loss
            running_val_loss = running_val_loss / len(val_loader)

        # Update the history
        history_loss.append(running_loss)
        history_val_loss.append(running_val_loss)

        # Update the progress bar
        pbar.set_postfix(
            {'loss': running_loss, 'val loss': running_val_loss})

    history = {'loss': history_loss, 'val_loss': history_val_loss}

    # Return the model and the history
    return net, history


def target_adaptation(device, net, input_10, input_20, spectral_ref, struct_ref, config):

    # Istanciate the optimizer
    optim = torch.optim.Adam(net.parameters(), lr=config.ta_learning_rate)

    # Move the model and the data to device
    net = net.to(device)
    input_10 = input_10.to(device)
    input_20 = input_20.to(device)
    spectral_ref = spectral_ref.to(device)
    struct_ref = struct_ref.to(device)
    spec_criterion = SpectralLoss().to(device)
    struct_criterion = StructLoss().to(device)

    # Initialize the history
    history_spec_loss = []
    history_struct_loss = []

    # Target adaptation loop

    # Set the model to train mode
    net.train()

    # Initialize the progress bar
    pbar = tqdm(range(config.ta_epochs))
    for epoch in pbar:

        # Zero the parameter gradients
        optim.zero_grad()

        # Forward step
        outputs = net(input_10, input_20)

        # Compute the losses
        spec_loss = spec_criterion(outputs, spectral_ref)
        struct_loss = struct_criterion(outputs, struct_ref)
        loss = config.lambda_1 * spec_loss + config.lambda_2 * struct_loss

        # Backward step
        loss.backward()

        # Weights update
        optim.step()

        # Update the history
        history_spec_loss.append(spec_loss.item())
        history_struct_loss.append(struct_loss.item())

        # Update the progress bar
        pbar.set_postfix(
            {'spec loss': spec_loss.item(), 'struct loss': struct_loss.item()})

    history = {'spec_loss': history_spec_loss, 'struct_loss': history_struct_loss}

    # Return the model and the history
    return net, history
