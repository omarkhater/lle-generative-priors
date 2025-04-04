import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import frameworks.LightenDiffusion.utils as utils

from frameworks.LightenDiffusion.models.unet import DiffusionUNet
from frameworks.LightenDiffusion.models.decom import CTDN


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Net(nn.Module):
    def __init__(self, 
                 mode='inference',
                 device='cuda',
                 beta_schedule='linear',
                 beta_start=0.0001,
                 beta_end=0.02,
                 num_diffusion_timesteps=1000,
                 num_sampling_timesteps=100,
                 in_channels=3,
                 out_channels=3,
                 ch=128,
                 ch_mult=(1, 2, 2, 2),
                 num_res_blocks=2,
                 dropout=0.1,
                 conditional=True,
                 resamp_with_conv=True,
                 stage1_path='ckpt/stage1'):
        """
        Initialize the diffusion model.
        
        Args:
            mode (str): 'training' or 'inference'
            device (str): Device to use ('cuda' or 'cpu')
            beta_schedule (str): Type of beta schedule ('linear', 'quad', etc.)
            beta_start (float): Starting value for beta schedule
            beta_end (float): Ending value for beta schedule
            num_diffusion_timesteps (int): Number of diffusion timesteps
            num_sampling_timesteps (int): Number of sampling timesteps
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            ch (int): Base channel count
            ch_mult (tuple): Channel multiplier for each resolution
            num_res_blocks (int): Number of residual blocks per resolution
            dropout (float): Dropout rate
            conditional (bool): Whether the model is conditional
            resamp_with_conv (bool): Whether to use convolution for resampling
            stage1_path (str): Path to stage1 weights
        """
        super(Net, self).__init__()

        self.mode = mode
        self.device = device
        self.conditional = conditional
        self.num_sampling_timesteps = num_sampling_timesteps
        self.num_diffusion_timesteps = num_diffusion_timesteps
        
        # Initialize UNet with explicit parameters
        self.Unet = DiffusionUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            conditional=conditional,
            resamp_with_conv=resamp_with_conv,
            device=device
        )
        
        if self.mode == 'training':
            self.decom = self.load_stage1(CTDN(), stage1_path)
        else:
            self.decom = CTDN()

        betas = get_beta_schedule(
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    @staticmethod
    def load_stage1(model, model_dir):
        checkpoint = utils.logging.load_checkpoint(os.path.join(model_dir, 'stage1_weight.pth.tar'), 'cuda')
        model.load_state_dict(checkpoint['model'], strict=True)
        return model

    def sample_training(self, x_cond, b, eta=0.):
        """
        Sample from the diffusion model during training or inference.
        
        Args:
            x_cond (Tensor): Low-light image features for conditioning
            b (Tensor): Beta schedule
            eta (float): Noise level parameter
            
        Returns:
            Tensor: Final prediction
        """
        skip = self.num_diffusion_timesteps // self.num_sampling_timesteps
        seq = range(0, self.num_diffusion_timesteps, skip)
        n, c, h, w = x_cond.shape
        seq_next = [-1] + list(seq[:-1])
        x = torch.randn(n, c, h, w, device=self.device)
        xs = [x]
        
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            
            # Pass condition as a separate parameter
            et = self.Unet(torch.cat([x_cond, xt], dim=1), t)
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(x.device))
            
        return xs[-1]

    def forward(self, x, y=None):
        """
        Forward pass that supports both paired (x, y) and unpaired (x) data.
        For paired data during training, x is the input and y is the ground truth.
        In inference mode (or if y is None), only x is used.
        """
        data_dict = {}
        b = self.betas.to(x.device)

        if self.training and y is not None:
            # Process paired inputs: use x for conditioning and y as ground truth
            output_x = self.decom(x, pred_fea=None)
            output_y = self.decom(y, pred_fea=None)

            # Extract features from x and y respectively
            low_R_x = output_x["low_R"]
            low_L_x = output_x["low_L"]
            low_fea_x = output_x["low_fea"]
            high_L_x = output_x["high_L"]

            low_R_y = output_y["low_R"]
            low_L_y = output_y["low_L"]

            # Compute conditioning feature from x and reference feature from y
            low_condition_norm = utils.data_transform(low_fea_x)
            reference_fea = low_R_y * torch.pow(low_L_y, 0.2)

            # Generate random timesteps for noise
            t = torch.randint(low=0, high=self.num_timesteps, size=(low_condition_norm.shape[0] // 2 + 1,)).to(self.device)
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:low_condition_norm.shape[0]].to(x.device)
            a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

            e = torch.randn_like(low_condition_norm)

            # Use x's features for input normalization
            high_input_norm = utils.data_transform(low_R_x * high_L_x)
            x_noise = high_input_norm * a.sqrt() + e * (1.0 - a).sqrt()

            # Noise estimation using the UNet
            noise_output = self.Unet(torch.cat([low_condition_norm, x_noise], dim=1), t.float())

            # Predict features using sampling
            pred_fea = self.sample_training(low_condition_norm, b)
            pred_fea = utils.inverse_data_transform(pred_fea)

            data_dict["noise_output"] = noise_output
            data_dict["e"] = e
            data_dict["pred_fea"] = pred_fea
            data_dict["reference_fea"] = reference_fea

        else:
            # In inference or unpaired mode, process x only.
            output = self.decom(x, pred_fea=None)
            low_fea = output["low_fea"]
            low_condition_norm = utils.data_transform(low_fea)

            pred_fea = self.sample_training(low_condition_norm, b)
            pred_fea = utils.inverse_data_transform(pred_fea)
            # Use the predicted features to obtain the final prediction
            pred_x = self.decom(x, pred_fea=pred_fea)["pred_img"]
            data_dict["pred_x"] = pred_x

        return data_dict


class DenoisingDiffusion(object):
    def __init__(self, 
                 mode='inference',
                 device='cuda',
                 ema_decay=0.9999,
                 beta_schedule='linear',
                 beta_start=0.0001,
                 beta_end=0.02,
                 num_diffusion_timesteps=1000,
                 num_sampling_timesteps=100,
                 in_channels=3,
                 out_channels=3,
                 ch=128,
                 ch_mult=(1, 2, 2, 2),
                 num_res_blocks=2,
                 dropout=0.1,
                 conditional=True,
                 resamp_with_conv=True,
                 stage1_path='ckpt/stage1',
                 ckpt_dir='ckpt',
                 optimizer_type='Adam',
                 optimizer_lr=2e-4,
                 optimizer_betas=(0.9, 0.999),
                 optimizer_weight_decay=0,
                 optimizer_eps=1e-8,
                 resume_path=None):
        """
        Initialize the denoising diffusion model.
        
        Args:
            mode (str): 'training' or 'inference'
            device (str): Device to use ('cuda' or 'cpu')
            ema_decay (float): EMA decay rate
            beta_schedule (str): Type of beta schedule ('linear', 'quad', etc.)
            beta_start (float): Starting value for beta schedule
            beta_end (float): Ending value for beta schedule
            num_diffusion_timesteps (int): Number of diffusion timesteps
            num_sampling_timesteps (int): Number of sampling timesteps
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            ch (int): Base channel count
            ch_mult (tuple): Channel multiplier for each resolution
            num_res_blocks (int): Number of residual blocks per resolution
            dropout (float): Dropout rate
            conditional (bool): Whether the model is conditional
            resamp_with_conv (bool): Whether to use convolution for resampling
            stage1_path (str): Path to stage1 weights
            ckpt_dir (str): Directory to save checkpoints
            optimizer_type (str): Type of optimizer ('Adam', 'AdamW', etc.)
            optimizer_lr (float): Learning rate for optimizer
            optimizer_betas (tuple): Betas for Adam optimizer
            optimizer_weight_decay (float): Weight decay for optimizer
            optimizer_eps (float): Epsilon for optimizer
            resume_path (str): Path to resume training from checkpoint
        """
        super().__init__()
        self.mode = mode
        self.device = device
        self.ckpt_dir = ckpt_dir
        self.resume_path = resume_path
        
        # Initialize the model with explicit parameters
        self.model = Net(
            mode=mode,
            device=device,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            num_diffusion_timesteps=num_diffusion_timesteps,
            num_sampling_timesteps=num_sampling_timesteps,
            in_channels=in_channels,
            out_channels=out_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            conditional=conditional,
            resamp_with_conv=resamp_with_conv,
            stage1_path=stage1_path
        )
        
        self.model.to(device)
        self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))

        self.ema_helper = EMAHelper(mu=ema_decay)
        self.ema_helper.register(self.model)

        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        
        # Create optimizer with explicit parameters
        self.optimizer = self._get_optimizer(
            optimizer_type=optimizer_type,
            lr=optimizer_lr,
            betas=optimizer_betas,
            weight_decay=optimizer_weight_decay,
            eps=optimizer_eps
        )
        
        self.start_epoch, self.step = 0, 0
        
        # Load checkpoint if resume path is provided
        if resume_path and os.path.isfile(resume_path):
            self.load_ddm_ckpt(resume_path)

    def _get_optimizer(self, optimizer_type='Adam', lr=2e-4, betas=(0.9, 0.999), 
                      weight_decay=0, eps=1e-8):
        """
        Create an optimizer for model parameters.
        
        Args:
            optimizer_type (str): Type of optimizer ('Adam', 'AdamW', etc.)
            lr (float): Learning rate
            betas (tuple): Beta parameters for Adam optimizer
            weight_decay (float): Weight decay parameter
            eps (float): Epsilon parameter
            
        Returns:
            torch.optim.Optimizer: The created optimizer
        """
        if optimizer_type == 'Adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
                eps=eps
            )
        elif optimizer_type == 'AdamW':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
                eps=eps
            )
        else:
            raise NotImplementedError(f"Optimizer {optimizer_type} not implemented")

    def load_ddm_ckpt(self, load_path, ema=False):
        """
        Load checkpoint from the given path.
        
        Args:
            load_path (str): Path to the checkpoint file
            ema (bool): Whether to apply EMA on the loaded model
        """
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        
        # Load optimizer and other training state if available
        if 'optimizer' in checkpoint and self.mode == 'training':
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'step' in checkpoint:
            self.step = checkpoint['step']
        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch']
        if 'ema_helper' in checkpoint:
            self.ema_helper.load_state_dict(checkpoint['ema_helper'])
            
        if ema:
            self.ema_helper.ema(self.model)
            
        print(f"=> loaded checkpoint {load_path} (step {self.step})")

    def train(self, train_loader, val_loader, n_epochs=100, validation_freq=1000, image_folder='results'):
        """
        Train the diffusion model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            n_epochs (int): Number of epochs to train
            validation_freq (int): Frequency of validation in steps
            image_folder (str): Folder to save validation images
        """
        cudnn.benchmark = True
            
        for name, param in self.model.named_parameters():
            if "decom" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        for epoch in range(self.start_epoch, n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                # Flatten the batch if the data has an extra dimension (e.g. multiple patches)
                if x.ndim == 5:
                    x = x.flatten(start_dim=0, end_dim=1)
                    y = y.flatten(start_dim=0, end_dim=1)
                self.model.train()
                self.step += 1

                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x, y)

                noise_loss, scc_loss = self.noise_estimation_loss(output)
                loss = noise_loss + scc_loss

                data_time += time.time() - data_start

                if self.step % 10 == 0:
                    print(f"step:{self.step}, noise_loss:{noise_loss.item():.5f} scc_loss:{scc_loss.item():.5f} time:{data_time / (i + 1):.5f}")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.step % validation_freq == 0 and self.step != 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step, image_folder)
                    utils.logging.save_checkpoint({
                        'step': self.step,
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict()
                    }, filename=os.path.join(self.ckpt_dir, 'model_latest'))

    def noise_estimation_loss(self, output):
        pred_fea, reference_fea = output["pred_fea"], output["reference_fea"]
        noise_output, e = output["noise_output"], output["e"]
        noise_loss = self.l2_loss(noise_output, e)
        scc_loss = 0.001 * self.l1_loss(pred_fea, reference_fea)
        return noise_loss, scc_loss

    def sample_validation_patches(self, val_loader, step, image_folder='results'):
        """
        Sample validation patches and save the results.
        
        Args:
            val_loader: DataLoader for validation data
            step (int): Current training step
            image_folder (str): Folder to save validation images
        """
        output_folder = os.path.join(image_folder, str(step))
        os.makedirs(output_folder, exist_ok=True)
        
        self.model.eval()

        with torch.no_grad():
            print(f'Performing validation at step: {step}')
            for i, (x, y) in enumerate(val_loader):
                b, _, img_h, img_w = x.shape

                img_h_64 = int(64 * np.ceil(img_h / 64.0))
                img_w_64 = int(64 * np.ceil(img_w / 64.0))
                x = F.pad(x, (0, img_w_64 - img_w, 0, img_h_64 - img_h), 'reflect')
                # For validation, we assume only x is used for inference.
                pred_x = self.model(x.to(self.device))["pred_x"][:, :, :img_h, :img_w]
                
                filename = y[0] if isinstance(y[0], str) else f"sample_{i}"
                utils.logging.save_image(pred_x, os.path.join(output_folder, filename))
