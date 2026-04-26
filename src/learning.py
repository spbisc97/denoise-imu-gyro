
import torch
import time
import matplotlib.pyplot as plt
plt.rcParams["legend.loc"] = "upper right"
plt.rcParams['axes.titlesize'] = 'x-large'
plt.rcParams['axes.labelsize'] = 'x-large'
plt.rcParams['legend.fontsize'] = 'x-large'
plt.rcParams['xtick.labelsize'] = 'x-large'
plt.rcParams['ytick.labelsize'] = 'x-large'
from termcolor import cprint
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from src.utils import pload, pdump, yload, ydump, mkdir, bmv
from src.utils import bmtm, bmtv, bmmt
from datetime import datetime
from src.lie_algebra import SO3, CPUSO3


class LearningBasedProcessing:
    def __init__(self, res_dir, tb_dir, net_class, net_params, address, dt):
        self.res_dir = res_dir
        self.tb_dir = tb_dir
        self.net_class = net_class
        self.net_params = net_params
        self._ready = False
        self.train_params = {}
        self.figsize = (20, 12)
        self.dt = dt # (s)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.show_plots = False
        self.address, self.tb_address = self.find_address(address)
        if address is None:  # create new address
            pdump(self.net_params, self.address, 'net_params.p')
            ydump(self.net_params, self.address, 'net_params.yaml')
        else:  # pick the network parameters
            self.net_params = pload(self.address, 'net_params.p')
            self.train_params = pload(self.address, 'train_params.p')
            self._ready = True
        self.path_weights = os.path.join(self.address, 'weights.pt')
        self.net = self.net_class(**self.net_params).to(self.device)
        if self._ready:  # fill network parameters
            self.load_weights()

    def find_address(self, address):
        """return path where net and training info are saved"""
        os.makedirs(self.res_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        if address == 'last':
            addresses = sorted(os.listdir(self.res_dir))
            if len(addresses) == 0:
                raise FileNotFoundError(
                    f"No runs found in {self.res_dir!r}. Train first or set address=None."
                )
            tb_address = os.path.join(self.tb_dir, str(len(addresses)))
            address = os.path.join(self.res_dir, addresses[-1])
        elif address is None:
            now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            address = os.path.join(self.res_dir, now)
            mkdir(address)
            tb_address = os.path.join(self.tb_dir, now)
        else:
            tb_address = None
        return address, tb_address

    def load_weights(self):
        weights = torch.load(self.path_weights, map_location=self.device)
        try:
            self.net.load_state_dict(weights)
        except RuntimeError as e:
            # Backward compatibility if model definition changed (e.g., new params).
            print(f"[WARN] Strict load failed ({e}); retrying with strict=False")
            missing, unexpected = self.net.load_state_dict(weights, strict=False)
            if missing:
                print(f"[WARN] Missing keys: {missing}")
            if unexpected:
                print(f"[WARN] Unexpected keys: {unexpected}")
        self.net.to(self.device)

    def build_optimizer(self, Optimizer, optimizer_params):
        """Create an optimizer with saner parameter groups for CNN + calibration."""
        params = dict(optimizer_params)
        base_lr = float(params.pop("lr"))
        base_weight_decay = float(params.pop("weight_decay", 0.0))
        calib_lr_scale = float(params.pop("calib_lr_scale", 1.0))
        calib_weight_decay = float(params.pop("calib_weight_decay", 0.0))
        norm_weight_decay = float(params.pop("norm_weight_decay", 0.0))

        decay_params = []
        norm_params = []
        calib_params = []
        no_decay_params = []

        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            if name in {"gyro_Rot", "gyro_bias", "dC", "b"}:
                calib_params.append(param)
            elif param.ndim <= 1 or name.endswith("bias"):
                no_decay_params.append(param)
            elif "bn" in name.lower() or "norm" in name.lower():
                norm_params.append(param)
            else:
                decay_params.append(param)

        param_groups = []
        if decay_params:
            param_groups.append({"params": decay_params, "lr": base_lr, "weight_decay": base_weight_decay})
        if norm_params:
            param_groups.append({"params": norm_params, "lr": base_lr, "weight_decay": norm_weight_decay})
        if no_decay_params:
            param_groups.append({"params": no_decay_params, "lr": base_lr, "weight_decay": 0.0})
        if calib_params:
            param_groups.append(
                {
                    "params": calib_params,
                    "lr": base_lr * calib_lr_scale,
                    "weight_decay": calib_weight_decay,
                }
            )
        return Optimizer(param_groups, lr=base_lr, **params)

    def train(self, dataset_class, dataset_params, train_params):
        """train the neural network. GPU is assumed"""
        self.train_params = train_params
        pdump(self.train_params, self.address, 'train_params.p')
        ydump(self.train_params, self.address, 'train_params.yaml')

        hparams = self.get_hparams(dataset_class, dataset_params, train_params)
        ydump(hparams, self.address, 'hparams.yaml')

        # define datasets
        dataset_train = dataset_class(**dataset_params, mode='train')
        dataset_train.init_train()
        dataset_val = dataset_class(**dataset_params, mode='val')
        dataset_val.init_val()

        # get class
        Optimizer = train_params['optimizer_class']
        Scheduler = train_params['scheduler_class']
        Loss = train_params['loss_class']

        # get parameters
        dataloader_params = train_params['dataloader']
        optimizer_params = dict(train_params['optimizer'])
        scheduler_params = train_params['scheduler']
        loss_params = train_params['loss']

        # define optimizer, scheduler and loss
        dataloader = DataLoader(dataset_train, **dataloader_params)
        optimizer = self.build_optimizer(Optimizer, optimizer_params)
        scheduler = Scheduler(optimizer, **scheduler_params)
        criterion = Loss(**loss_params).to(self.device)

        try:
            scheduler_name = getattr(Scheduler, "__name__", str(Scheduler))
        except Exception:
            scheduler_name = str(Scheduler)
        try:
            init_lrs = [pg.get("lr", None) for pg in optimizer.param_groups]
        except Exception:
            init_lrs = []
        print(f"[INFO] Scheduler: {scheduler_name} params={scheduler_params}")
        if init_lrs:
            print(f"[INFO] Initial LR(s): {init_lrs}")

        # remaining training parameters
        freq_val = train_params['freq_val']
        n_epochs = train_params['n_epochs']

        # init net w.r.t dataset
        mean_u, std_u = dataset_train.mean_u, dataset_train.std_u
        self.net.set_normalized_factors(mean_u, std_u)

        # start tensorboard writer
        writer = SummaryWriter(self.tb_address)
        start_time = time.time()
        best_loss = torch.tensor(float('inf'))

        # define some function for seeing evolution of training
        def write(epoch, loss_epoch):
            lr_groups = [pg["lr"] for pg in optimizer.param_groups]
            writer.add_scalar('loss/train', loss_epoch.item(), epoch)
            writer.add_scalar('lr', lr_groups[0], epoch)
            print('Train Epoch: {:2d} \tLoss: {:.4f} \tLR: {}'.format(
                epoch, loss_epoch.item(), lr_groups))

        def write_time(epoch, start_time):
            delta_t = time.time() - start_time
            print("Amount of time spent for epochs " +
                "{}-{}: {:.1f}s\n".format(epoch - freq_val, epoch, delta_t))
            writer.add_scalar('time_spend', delta_t, epoch)

        def write_val(loss, best_loss):
            if not torch.isfinite(loss):
                msg = 'validation loss is not finite (NaN/Inf)'
                cprint(msg, 'yellow')
            elif loss <= best_loss:
                msg = 'validation loss decreases! :) '
                msg += '(curr/prev loss {:.4f}/{:.4f})'.format(loss.item(),
                    best_loss.item())
                cprint(msg, 'green')
                best_loss = loss
                self.save_net()
            else:
                msg = 'validation loss increases! :( '
                msg += '(curr/prev loss {:.4f}/{:.4f})'.format(loss.item(),
                    best_loss.item())
                cprint(msg, 'yellow')
            writer.add_scalar('loss/val', loss.item(), epoch)
            return best_loss

        def validate_initial_model():
            initial_loss = self.loop_val(dataset_val, criterion)
            writer.add_scalar('loss/val', initial_loss.item(), 0)
            if torch.isfinite(initial_loss):
                cprint(
                    'initial validation loss {:.4f} saved as starting checkpoint'.format(initial_loss.item()),
                    'cyan',
                )
                self.save_net()
                return initial_loss
            cprint('initial validation loss is not finite; starting without checkpoint', 'yellow')
            return torch.tensor(float('inf'))

        best_loss = validate_initial_model()

        # training loop !
        for epoch in range(1, n_epochs + 1):
            loss_epoch = self.loop_train(dataloader, optimizer, criterion)
            write(epoch, loss_epoch)
            scheduler.step()
            if epoch % freq_val == 0:
                loss = self.loop_val(dataset_val, criterion)
                write_time(epoch, start_time)
                best_loss = write_val(loss, best_loss)
                start_time = time.time()
        # training is over !

        if not os.path.exists(self.path_weights):
            # Ensure there is at least one set of weights to load for testing.
            self.save_net()

        # test on new data
        dataset_test = dataset_class(**dataset_params, mode='test')
        self.load_weights()
        test_loss = self.loop_val(dataset_test, criterion)
        dict_loss = {
            'final_loss/val': best_loss.item(),
            'final_loss/test': test_loss.item()
            }
        writer.add_hparams(hparams, dict_loss)
        ydump(dict_loss, self.address, 'final_loss.yaml')
        writer.close()

    def loop_train(self, dataloader, optimizer, criterion):
        """Forward-backward loop over training data"""
        loss_epoch = 0
        optimizer.zero_grad()
        for us, xs in dataloader:
            us = dataloader.dataset.add_noise(us.to(self.device))
            hat_xs = self.net(us)
            loss = criterion(xs.to(self.device), hat_xs)/len(dataloader)
            loss.backward()
            loss_epoch += loss.detach().cpu()
        optimizer.step()
        return loss_epoch

    def loop_val(self, dataset, criterion):
        """Forward loop over validation data"""
        loss_epoch = 0
        self.net.eval()
        with torch.no_grad():
            for i in range(len(dataset)):
                us, xs = dataset[i]
                hat_xs = self.net(us.to(self.device).unsqueeze(0))
                loss = criterion(xs.to(self.device).unsqueeze(0), hat_xs)/len(dataset)
                loss_epoch += loss.cpu()
        self.net.train()
        return loss_epoch

    def save_net(self):
        """save the weights on the net in CPU"""
        self.net.eval().cpu()
        torch.save(self.net.state_dict(), self.path_weights)
        self.net.train().to(self.device)

    def get_hparams(self, dataset_class, dataset_params, train_params):
        """return all training hyperparameters in a dict"""
        Optimizer = train_params['optimizer_class']
        Scheduler = train_params['scheduler_class']
        Loss = train_params['loss_class']

        # get training class parameters
        dataloader_params = train_params['dataloader']
        optimizer_params = train_params['optimizer']
        scheduler_params = train_params['scheduler']
        loss_params = train_params['loss']

        # remaining training parameters
        freq_val = train_params['freq_val']
        n_epochs = train_params['n_epochs']

        dict_class = {
            'Optimizer': str(Optimizer),
            'Scheduler': str(Scheduler),
            'Loss': str(Loss)
        }

        return {**dict_class, **dataloader_params, **optimizer_params,
                **loss_params, **scheduler_params,
                'n_epochs': n_epochs, 'freq_val': freq_val}

    def test(self, dataset_class, dataset_params, modes):
        """test a network once training is over"""
        self.dataset_class = dataset_class
        self.dataset_params = dataset_params

        # get loss function
        Loss = self.train_params['loss_class']
        loss_params = self.train_params['loss']
        criterion = Loss(**loss_params).to(self.device)

        # test on each type of sequence
        for mode in modes:
            dataset = dataset_class(**dataset_params, mode=mode)
            self.loop_test(dataset, criterion)
            self.display_test(dataset, mode)

    def loop_test(self, dataset, criterion):
        """Forward loop over test data"""
        self.net.eval()
        for i in range(len(dataset)):
            seq = dataset.sequences[i]
            us, xs = dataset[i]
            with torch.no_grad():
                hat_xs = self.net(us.to(self.device).unsqueeze(0))
            loss = criterion(xs.to(self.device).unsqueeze(0), hat_xs)
            mkdir(self.address, seq)
            mondict = {
                'hat_xs': hat_xs[0].cpu(),
                'loss': loss.cpu().item(),
            }
            pdump(mondict, self.address, seq, 'results.p')

    def display_test(self, dataset, mode):
        raise NotImplementedError


class GyroLearningBasedProcessing(LearningBasedProcessing):
    def __init__(self, res_dir, tb_dir, net_class, net_params, address, dt):
        super().__init__(res_dir, tb_dir, net_class, net_params, address, dt)
        self.roe_dist = [7, 14, 21, 28, 35] # m
        self.freq = 100 # subsampling frequency for RTE computation
        self.roes = { # relative trajectory errors
            'Rots': [],
            'yaws': [],
            }
        self.gyro_biases = [4.5,1.2,-0.2] # deg/s
        self.enable_calibrated_imu_baseline = False
        self.calib_baseline_steps = 300
        self.calib_baseline_lr = 5e-2
        self.init_from_calib_baseline = False
        self.freeze_calib_params = False
        self.calib_source = "static"
        self.static_calib_path = None

    def apply_calib_to_net(self, calib):
        """Initialize the net's static calibration parameters from a calib dict."""
        with torch.no_grad():
            if hasattr(self.net, "gyro_Rot") and "dC" in calib:
                self.net.gyro_Rot.copy_(calib["dC"].to(self.net.gyro_Rot.device))
            if hasattr(self.net, "gyro_bias") and "b" in calib:
                self.net.gyro_bias.copy_(calib["b"].to(self.net.gyro_bias.device))

        if self.freeze_calib_params:
            if hasattr(self.net, "gyro_Rot"):
                self.net.gyro_Rot.requires_grad_(False)
            if hasattr(self.net, "gyro_bias"):
                self.net.gyro_bias.requires_grad_(False)

    @staticmethod
    def _calib_to_yaml(calib):
        return {
            "dC": calib["dC"].detach().cpu().tolist(),
            "b": calib["b"].detach().cpu().tolist(),
        }

    @staticmethod
    def _calib_from_yaml(path):
        data = yload(path)
        return {
            "dC": torch.tensor(data["dC"], dtype=torch.float32),
            "b": torch.tensor(data["b"], dtype=torch.float32),
        }

    def load_or_fit_static_calib(self, dataset_class, dataset_params, train_params):
        if not self.static_calib_path:
            raise ValueError("static_calib_path is required when calib_source='static'")

        if os.path.isfile(self.static_calib_path):
            print(f"[calib] loading static calibration: {self.static_calib_path}")
            return self._calib_from_yaml(self.static_calib_path)

        print(f"[calib] static calibration not found, fitting once: {self.static_calib_path}")
        calib = self.fit_calibrated_imu(
            dataset_class,
            dataset_params,
            train_params,
            n_steps=int(self.calib_baseline_steps),
            lr=float(self.calib_baseline_lr),
        )
        ydump(self._calib_to_yaml(calib), self.static_calib_path)
        print(f"[calib] saved static calibration: {self.static_calib_path}")
        return calib

    def compute_static_calibration(self, dataset_class, dataset_params, train_params):
        calib = self.fit_calibrated_imu(
            dataset_class,
            dataset_params,
            train_params,
            n_steps=int(self.calib_baseline_steps),
            lr=float(self.calib_baseline_lr),
        )
        if self.static_calib_path:
            ydump(self._calib_to_yaml(calib), self.static_calib_path)
            print(f"[calib] saved static calibration: {self.static_calib_path}")
        return calib

    def train(self, dataset_class, dataset_params, train_params):
        if self.init_from_calib_baseline:
            if self.calib_source == "static":
                calib = self.load_or_fit_static_calib(dataset_class, dataset_params, train_params)
            elif self.calib_source == "fit":
                calib = self.compute_static_calibration(dataset_class, dataset_params, train_params)
            elif self.calib_source == "none":
                calib = None
            else:
                raise ValueError(f"Unknown calib_source={self.calib_source!r}")
            if calib is None:
                self.freeze_calib_params = False
            else:
                ydump(self._calib_to_yaml(calib), self.address, "static_calibration_used.yaml")
                self.apply_calib_to_net(calib)
        return super().train(dataset_class, dataset_params, train_params)

    def _calibration_windows(self, dataset, *, sequences, window_size, stride, target):
        us_windows = []
        xs_windows = []
        mask_target = "mask" in str(target)

        for seq in sequences:
            data = pload(dataset.predata_dir, seq + ".p")
            us = data["us"]
            xs = data["xs"]
            n_max = min(us.shape[0], xs.shape[0])
            if n_max < window_size:
                continue

            starts = list(range(0, n_max - window_size + 1, stride))
            tail_start = n_max - window_size
            if starts[-1] != tail_start:
                starts.append(tail_start)

            for start in starts:
                end = min(start + window_size, n_max)
                u = us[start:end]
                x = xs[start:end]
                if u.shape[0] < dataset.max_train_freq:
                    continue
                usable = u.shape[0] - (u.shape[0] % dataset.max_train_freq)
                u = u[:usable]
                x = x[:usable]
                if mask_target and x.shape[-1] > 3 and torch.count_nonzero(x[:, 3] > 0) == 0:
                    continue
                us_windows.append(u)
                xs_windows.append(x)

        if not us_windows:
            raise ValueError("No valid calibration windows were built from the selected sequences.")
        return TensorDataset(torch.stack(us_windows), torch.stack(xs_windows))

    @staticmethod
    def _evaluate_calib_loss(model, criterion, dataloader, device):
        model.eval()
        values = []
        with torch.no_grad():
            for us, xs in dataloader:
                us = us.to(device)
                xs = xs.to(device)
                loss = criterion(xs, model(us))
                value = float(loss.detach().cpu())
                if np.isfinite(value):
                    values.append(value)
        model.train()
        return float(np.mean(values)) if values else float("inf")

    def fit_calibrated_imu(self, dataset_class, dataset_params, train_params, *, n_steps=300, lr=5e-2):
        """
        Fit the 'calibrated IMU (prop.)' baseline from the paper by optimizing a
        static correction:
            ω_hat = (I + dC) ω + b
        on training segments using the same GyroLoss.
        """
        from src.networks import CalibratedIMUNet

        out_path = os.path.join(self.address, "calibrated_imu.p")
        out_yaml = os.path.join(self.address, "calibrated_imu.yaml")
        report_yaml = os.path.join(self.address, "calibrated_imu_report.yaml")
        if os.path.exists(out_path):
            return pload(out_path)

        # Build deterministic calibration windows. Calibration is run rarely, so
        # prefer repeatability and full coverage over stochastic speed.
        dataset_train = dataset_class(**dataset_params, mode='train')
        target = train_params["loss"].get("target", "")
        window_size = int(dataset_params.get("N", dataset_train.N))
        stride = max(dataset_train.max_train_freq, window_size // 2)
        train_windows = self._calibration_windows(
            dataset_train,
            sequences=dataset_train.train_sequences,
            window_size=window_size,
            stride=stride,
            target=target,
        )

        dataloader_params = dict(train_params.get('dataloader', {}))
        dataloader_params.setdefault('batch_size', 10)
        dataloader_params["shuffle"] = False
        dataloader_params.setdefault('num_workers', 0)
        dataloader_params.setdefault('pin_memory', False)
        train_loader = DataLoader(train_windows, **dataloader_params)

        val_loader = None
        val_sequences = list(getattr(dataset_train, "val_sequences", []))
        if val_sequences:
            val_windows = self._calibration_windows(
                dataset_train,
                sequences=val_sequences,
                window_size=window_size,
                stride=stride,
                target=target,
            )
            val_loader = DataLoader(val_windows, **dataloader_params)

        Loss = train_params['loss_class']
        loss_params = train_params['loss']
        criterion = Loss(**loss_params).to(self.device)

        model = CalibratedIMUNet().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(n_steps)),
            eta_min=max(float(lr) * 0.02, 1e-5),
        )

        best_loss = float("inf")
        best_state = None
        best_step = 0
        last_train_loss = float("inf")
        eval_every = max(1, min(50, int(n_steps)))
        model.train()
        step = 0
        while step < n_steps:
            for us, xs in train_loader:
                step += 1
                # Calibrated-IMU baseline optimizes static parameters; do not
                # inject artificial noise here to keep it comparable to eval.
                us = us.to(self.device)
                xs = xs.to(self.device)
                hat_xs = model(us)
                loss = criterion(xs, hat_xs)
                if not torch.isfinite(loss) or not loss.requires_grad:
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                last_train_loss = float(loss.detach().cpu())

                if step % eval_every == 0 or step == n_steps:
                    score = (
                        self._evaluate_calib_loss(model, criterion, val_loader, self.device)
                        if val_loader is not None
                        else self._evaluate_calib_loss(model, criterion, train_loader, self.device)
                    )
                    if score < best_loss:
                        best_loss = score
                        best_step = step
                        best_state = {
                            "dC": model.dC.detach().cpu().clone(),
                            "b": model.b.detach().cpu().clone(),
                        }
                    print(
                        f"[calib] step={step}/{n_steps} "
                        f"train_loss={last_train_loss:.6f} score={score:.6f} best={best_loss:.6f}"
                    )
                if step >= n_steps:
                    break

        if best_state is not None:
            with torch.no_grad():
                model.dC.copy_(best_state["dC"].to(model.dC.device))
                model.b.copy_(best_state["b"].to(model.b.device))

        calib = {
            "dC": model.dC.detach().cpu(),
            "b": model.b.detach().cpu(),
        }
        pdump(calib, out_path)
        ydump(self._calib_to_yaml(calib), out_yaml)
        ydump(
            {
                "best_step": best_step,
                "best_score": best_loss,
                "final_train_loss": last_train_loss,
                "n_steps": int(n_steps),
                "lr": float(lr),
                "window_size": window_size,
                "stride": stride,
                "train_windows": len(train_windows),
                "val_windows": len(val_loader.dataset) if val_loader is not None else 0,
                **self._calib_to_yaml(calib),
            },
            report_yaml,
        )
        return calib

    def display_test(self, dataset, mode):
        self.roes = {
            'Rots': [],
            'yaws': [],
        }
        calib = None
        if getattr(self, "enable_calibrated_imu_baseline", False):
            if self.calib_source == "static":
                calib = self.load_or_fit_static_calib(
                    self.dataset_class,
                    self.dataset_params,
                    self.train_params,
                )
            elif self.calib_source == "fit":
                calib = self.fit_calibrated_imu(
                    self.dataset_class,
                    self.dataset_params,
                    self.train_params,
                    n_steps=int(self.calib_baseline_steps),
                    lr=float(self.calib_baseline_lr),
                )
            elif self.calib_source == "none":
                calib = None
            else:
                raise ValueError(f"Unknown calib_source={self.calib_source!r}")

        self.to_open_vins(dataset)
        for i, seq in enumerate(dataset.sequences):
            print('\n', 'Results for sequence ' + seq )
            self.seq = seq
            # get ground truth
            self.gt = dataset.load_gt(i)
            Rots = SO3.from_quaternion(self.gt['qs'].to(self.device))
            self.gt['Rots'] = Rots.cpu()
            self.gt['rpys'] = SO3.to_rpy(Rots).cpu()
            # get data and estimate
            self.net_us = pload(self.address, seq, 'results.p')['hat_xs']
            self.raw_us, _ = dataset[i]
            N = self.net_us.shape[0]
            self.gyro_corrections =  (self.raw_us[:, :3] - self.net_us[:, :3])
            self.ts = torch.linspace(0, N*self.dt, N)

            self.convert()
            self.plot_gyro(calib=calib)
            self.plot_gyro_correction()
            if self.show_plots:
                plt.show()

    def to_open_vins(self, dataset):
        """
        Export results to Open-VINS format. Use them eval toolbox available 
        at https://github.com/rpng/open_vins/
        """

        for i, seq in enumerate(dataset.sequences):
            self.seq = seq
            # get ground truth
            self.gt = dataset.load_gt(i)
            raw_us, _ = dataset[i]
            net_us = pload(self.address, seq, 'results.p')['hat_xs']
            N = net_us.shape[0]
            net_qs, imu_Rots, net_Rots = self.integrate_with_quaternions_superfast(N, raw_us, net_us)
            path = os.path.join(self.address, seq + '.txt')
            header = "timestamp(s) tx ty tz qx qy qz qw"
            x = np.zeros((net_qs.shape[0], 8))
            x[:, 0] = self.gt['ts'][:net_qs.shape[0]]
            x[:, [7, 4, 5, 6]] = net_qs
            np.savetxt(path, x[::10], header=header, delimiter=" ",
                    fmt='%1.9f')

    def convert(self):
        # s -> min
        l = 1/60
        self.ts *= l

        # rad -> deg
        l = 180/np.pi
        self.gyro_corrections *= l
        self.gt['rpys'] *= l

    def integrate_with_quaternions_superfast(self, N, raw_us, net_us, cal_us=None):
        imu_qs = SO3.qnorm(SO3.qexp(raw_us[:, :3].to(self.device).double()*self.dt))
        net_qs = SO3.qnorm(SO3.qexp(net_us[:, :3].to(self.device).double()*self.dt))
        if cal_us is not None:
            cal_us = SO3.qnorm(SO3.qexp(cal_us[:, :3].to(self.device).double()*self.dt))
        
        Rot0 = SO3.qnorm(self.gt['qs'][:2].to(self.device).double())
        imu_qs[0] = Rot0[0]
        net_qs[0] = Rot0[0]
        if cal_us is not None:
            cal_us[0] = Rot0[0]

        N = np.log2(imu_qs.shape[0])
        for i in range(int(N)):
            k = 2**i
            imu_qs[k:] = SO3.qnorm(SO3.qmul(imu_qs[:-k], imu_qs[k:]))
            net_qs[k:] = SO3.qnorm(SO3.qmul(net_qs[:-k], net_qs[k:]))
            if cal_us is not None:
                cal_us[k:] = SO3.qnorm(SO3.qmul(cal_us[:-k], cal_us[k:]))

        if int(N) < N:
            k = 2**int(N)
            k2 = imu_qs[k:].shape[0]
            imu_qs[k:] = SO3.qnorm(SO3.qmul(imu_qs[:k2], imu_qs[k:]))
            net_qs[k:] = SO3.qnorm(SO3.qmul(net_qs[:k2], net_qs[k:]))
            if cal_us is not None:
                cal_us[k:] = SO3.qnorm(SO3.qmul(cal_us[:k2], cal_us[k:]))
        
        if cal_us is not None:
            return net_qs.cpu(), SO3.from_quaternion(imu_qs).float(), SO3.from_quaternion(net_qs).float(), SO3.from_quaternion(cal_us).float()
        imu_Rots = SO3.from_quaternion(imu_qs).float()
        net_Rots = SO3.from_quaternion(net_qs).float()
        return net_qs.cpu(), imu_Rots, net_Rots

    def plot_gyro(self, calib=None):
        N = self.raw_us.shape[0]
        raw_us = self.raw_us[:, :3]
        net_us = self.net_us[:, :3] 
        cal_us = None
        if calib is not None:
            dC = calib["dC"].to(raw_us.device)
            b = calib["b"].to(raw_us.device)
            C = torch.eye(3, device=raw_us.device, dtype=raw_us.dtype) + dC.to(raw_us.dtype)
            cal_us = (raw_us @ C.T) + b.view(1, 3)

        if cal_us is not None:
            _, imu_Rots, net_Rots, cal_Rots = self.integrate_with_quaternions_superfast(N, raw_us, net_us, cal_us)
        else:
            _, imu_Rots, net_Rots = self.integrate_with_quaternions_superfast(N, raw_us, net_us)

        imu_rpys = 180/np.pi*SO3.to_rpy(imu_Rots).cpu()
        net_rpys = 180/np.pi*SO3.to_rpy(net_Rots).cpu()
        cal_rpys = 180/np.pi*SO3.to_rpy(cal_Rots).cpu() if cal_us is not None else None
        self.plot_orientation(imu_rpys, net_rpys, N, cal_rpys=cal_rpys)
        self.plot_orientation_error(imu_Rots, net_Rots, N, cal_Rots=cal_Rots if cal_us is not None else None)

    def plot_orientation(self, imu_rpys, net_rpys, N, cal_rpys=None):
        title = "Orientation estimation"
        gt = self.gt['rpys'][:N]
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        axs[0].set(ylabel='roll (deg)', title=title)
        axs[1].set(ylabel='pitch (deg)')
        axs[2].set(xlabel='$t$ (min)', ylabel='yaw (deg)')

        for i in range(3):
            axs[i].plot(self.ts, gt[:, i], color='black', label=r'Reference')
            axs[i].plot(self.ts, imu_rpys[:, i], color='red', label=r'Std IMU')
            axs[i].plot(self.ts, net_rpys[:, i], color='blue', label=r'LLyte IMU')
            if cal_rpys is not None:
                axs[i].plot(self.ts, cal_rpys[:, i], color='green', label=r'cal IMU')
            axs[i].set_xlim(self.ts[0], self.ts[-1])
        self.savefig(axs, fig, 'orientation')

    def plot_orientation_error(self, imu_Rots, net_Rots, N,cal_Rots=None):
        gt = self.gt['Rots'][:N].to(self.device)
        raw_err = 180/np.pi*SO3.log(bmtm(imu_Rots, gt)).cpu()
        net_err = 180/np.pi*SO3.log(bmtm(net_Rots, gt)).cpu()
        if cal_Rots is not None:
            cal_err = 180/np.pi*SO3.log(bmtm(cal_Rots, gt)).cpu()
        title = "$SO(3)$ orientation error"
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        axs[0].set(ylabel='roll (deg)', title=title)
        axs[1].set(ylabel='pitch (deg)')
        axs[2].set(xlabel='$t$ (min)', ylabel='yaw (deg)')

        for i in range(3):
            axs[i].plot(self.ts, raw_err[:, i], color='red', label=r'Std IMU')
            axs[i].plot(self.ts, net_err[:, i], color='blue', label=r'LLyte IMU')
            if cal_Rots is not None:
                axs[i].plot(self.ts, cal_err[:, i], color='green', label=r'cal IMU')
            axs[i].plot(self.ts, torch.zeros(N), color='black', linestyle='--')
            axs[i].set_ylim(-10, 10)
            axs[i].set_xlim(self.ts[0], self.ts[-1])
        self.savefig(axs, fig, 'orientation_error')

    def plot_gyro_correction(self):
        title = "Gyro correction" + self.end_title
        ylabel = 'gyro correction (deg/s)'
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set(xlabel='$t$ (min)', ylabel=ylabel, title=title)
        plt.plot(self.ts, self.gyro_corrections, label=r'LLyte Correction')
        ax.set_xlim(self.ts[0], self.ts[-1])
        self.savefig(ax, fig, 'gyro_correction')

    @property
    def end_title(self):
        return " for sequence " + self.seq.replace("_", " ")

    def savefig(self, axs, fig, name):
        if isinstance(axs, np.ndarray):
            for i in range(len(axs)):
                axs[i].grid()
                axs[i].legend()
        else:
            axs.grid()
            axs.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.address, self.seq, name + '.png'))
