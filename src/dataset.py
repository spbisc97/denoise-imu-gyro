from src.utils import pdump, pload, bmtv, bmtm
from src.lie_algebra import SO3
from termcolor import cprint
from torch.utils.data.dataset import Dataset
from scipy.interpolate import interp1d
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import torch
import sys

BLACKBIRD_IMU_USECOLS = [0, 10, 11, 12, 14, 15, 16]
BLACKBIRD_STATE_USECOLS = [0, 5, 6, 7, 8, 9, 10, 11]
BLACKBIRD_BODY_TO_IMU_QUAT_WXYZ = torch.tensor(
    [0.707479362748676, 0.002029830683065, -0.007745228390866, 0.706688646087723],
    dtype=torch.float64,
)


def _ensure_2d(array):
    array = np.asarray(array)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array


def _to_seconds(ts):
    ts = np.asarray(ts, dtype=np.float64)
    if ts.size < 2:
        return ts
    diffs = np.diff(ts[: min(ts.shape[0], 1024)])
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return ts
    median_dt = np.median(diffs)
    if median_dt > 1e6:
        return ts / 1e9
    if median_dt > 1e3:
        return ts / 1e6
    return ts


def _load_blackbird_csv(path, usecols, *, skip_header=1):
    data = np.genfromtxt(path, delimiter=",", skip_header=skip_header, usecols=usecols, dtype=np.float64)
    data = _ensure_2d(data)
    data = data[np.isfinite(data[:, 0])]
    if data.size == 0:
        raise ValueError(f"No numeric rows found in {path!r}")
    return data

class BaseDataset(Dataset):

    def __init__(self, predata_dir, train_seqs, val_seqs, test_seqs, mode, N,
        min_train_freq=128, max_train_freq=512, dt=0.005,
        train_windows_per_seq=16):
        super().__init__()
        # where record pre loaded data
        self.predata_dir = predata_dir
        os.makedirs(self.predata_dir, exist_ok=True)

        self.mode = mode
        self.train_sequences = list(train_seqs)
        self.val_sequences = list(val_seqs)
        self.test_sequences = list(test_seqs)
        self.all_requested_sequences = list(dict.fromkeys(
            self.train_sequences + self.val_sequences + self.test_sequences
        ))
        self.path_normalize_factors = self._normalization_cache_path(self.train_sequences)
        # choose between training, validation or test sequences
        _, self.sequences = self.get_sequences(self.train_sequences, self.val_sequences,
            self.test_sequences)
        self.mean_u = 0
        self.std_u = 1
        self.mode = mode  # train, val or test
        self._train = False
        self._val = False
        # noise density
        self.imu_std = torch.Tensor([8e-5, 1e-3]).float()
        # bias repeatability (without in-run bias stability)
        self.imu_b0 = torch.Tensor([1e-3, 1e-3]).float()
        # IMU sampling time
        self.dt = dt # (s)
        # sequence size during training
        self.N = N # power of 2
        self.min_train_freq = min_train_freq
        self.max_train_freq = max_train_freq
        self.train_windows_per_seq = int(train_windows_per_seq)
        self.uni = torch.distributions.uniform.Uniform(-torch.ones(1),
            torch.ones(1))

    def get_sequences(self, train_seqs, val_seqs, test_seqs):
        """Choose sequence list depending on dataset mode"""
        sequences_dict = {
            'train': train_seqs,
            'val': val_seqs,
            'test': test_seqs,
        }
        return sequences_dict['train'], sequences_dict[self.mode]

    def __getitem__(self, i):
        if self._train and len(self.sequences) > 0:
            i = i % len(self.sequences)
        mondict = self.load_seq(i)
        N_max = mondict['xs'].shape[0]
        if self._train: # random start
            if N_max < self.N:
                raise ValueError(
                    f"Sequence {self.sequences[i]!r} is shorter than the requested training window "
                    f"({N_max} < {self.N})."
                )
            max_start = N_max - self.N
            n0 = int(torch.randint(0, max_start + 1, (1, )).item()) if max_start > 0 else 0
            nend = n0 + self.N
        elif self._val: # end sequence
            n0 = self.max_train_freq + self.N
            nend = N_max - ((N_max - n0) % self.max_train_freq)
        else:  # full sequence
            n0 = 0
            nend = N_max - (N_max % self.max_train_freq)
        u = mondict['us'][n0: nend]
        x = mondict['xs'][n0: nend]
        return u, x

    def __len__(self):
        if self._train:
            return len(self.sequences) * self.train_windows_per_seq
        return len(self.sequences)

    def add_noise(self, u):
        """Add Gaussian noise and bias to input"""
        noise = torch.randn_like(u)
        noise[:, :, :3] = noise[:, :, :3] * self.imu_std[0]
        noise[:, :, 3:6] = noise[:, :, 3:6] * self.imu_std[1]

        # bias repeatability (without in run bias stability)
        b0 = torch.empty(u.shape[0], u.shape[2], device=u.device, dtype=u.dtype)
        b0.uniform_(-1.0, 1.0)
        b0[:, :3] = b0[:, :3] * self.imu_b0[0]
        b0[:, 3:6] = b0[:, 3:6] * self.imu_b0[1]
        u = u + noise + b0.unsqueeze(1)
        return u

    def init_train(self):
        self._train = True
        self._val = False

    def init_val(self):
        self._train = False
        self._val = True

    def finalize_preprocessing(self):
        self.mean_u, self.std_u = self.init_normalize_factors(self.train_sequences)

    def length(self):
        return self._length

    def load_seq(self, i):
        return pload(self.predata_dir, self.sequences[i] + '.p')

    def load_gt(self, i):
        return pload(self.predata_dir, self.sequences[i] + '_gt.p')

    def _normalization_cache_path(self, train_seqs):
        signature = ",".join(train_seqs) if train_seqs else "__empty__"
        digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
        return os.path.join(self.predata_dir, f"nf_{digest}.p")

    def init_normalize_factors(self, train_seqs):
        if os.path.exists(self.path_normalize_factors):
            mondict = pload(self.path_normalize_factors)
            return mondict['mean_u'], mondict['std_u']

        path = os.path.join(self.predata_dir, train_seqs[0] + '.p')
        if not os.path.exists(path):
            print("init_normalize_factors not computed")
            return 0, 0

        print('Start computing normalizing factors ...')
        cprint("Do it only on training sequences, it is vital!", 'yellow')
        # first compute mean
        num_data = 0

        for i, sequence in enumerate(train_seqs):
            pickle_dict = pload(self.predata_dir, sequence + '.p')
            us = pickle_dict['us']
            sms = pickle_dict['xs']
            if i == 0:
                mean_u = us.sum(dim=0)
                num_positive = sms.sum(dim=0)
                num_negative = sms.shape[0] - sms.sum(dim=0)
            else:
                mean_u += us.sum(dim=0)
                num_positive += sms.sum(dim=0)
                num_negative += sms.shape[0] - sms.sum(dim=0)
            num_data += us.shape[0]
        mean_u = mean_u / num_data
        pos_weight = num_negative / num_positive

        # second compute standard deviation
        for i, sequence in enumerate(train_seqs):
            pickle_dict = pload(self.predata_dir, sequence + '.p')
            us = pickle_dict['us']
            if i == 0:
                std_u = ((us - mean_u) ** 2).sum(dim=0)
            else:
                std_u += ((us - mean_u) ** 2).sum(dim=0)
        std_u = (std_u / num_data).sqrt()
        normalize_factors = {
            'mean_u': mean_u,
            'std_u': std_u,
        }
        print('... ended computing normalizing factors')
        print('pos_weight:', pos_weight)
        print('This values most be a training parameters !')
        print('mean_u    :', mean_u)
        print('std_u     :', std_u)
        print('num_data  :', num_data)
        pdump(normalize_factors, self.path_normalize_factors)
        return mean_u, std_u

    def read_data(self, data_dir):
        raise NotImplementedError

    @staticmethod
    def interpolate(x, t, t_int):
            """
            Interpolate ground truth at the sensor timestamps
            """

            # vector interpolation
            x_int = np.zeros((t_int.shape[0], x.shape[1]))
            for i in range(x.shape[1]):
                if i in [4, 5, 6, 7]:
                    continue
                x_int[:, i] = np.interp(t_int, t, x[:, i])
            # quaternion interpolation
            t_int = torch.Tensor(t_int - t[0])
            t = torch.Tensor(t - t[0])
            qs = SO3.qnorm(torch.Tensor(x[:, 4:8]))
            x_int[:, 4:8] = SO3.qinterp(qs, t, t_int).numpy()
            return x_int


class EUROCDataset(BaseDataset):
    """
        Dataloader for the EUROC Data Set.
    """

    def __init__(self, data_dir, predata_dir, train_seqs, val_seqs,
                test_seqs, mode, N, min_train_freq, max_train_freq, dt=0.005,
                train_windows_per_seq=16):
        super().__init__(
            predata_dir,
            train_seqs,
            val_seqs,
            test_seqs,
            mode,
            N,
            min_train_freq,
            max_train_freq,
            dt,
            train_windows_per_seq,
        )
        # convert raw data to pre loaded data
        self.read_data(data_dir)
        self.finalize_preprocessing()

    def read_data(self, data_dir):
        r"""Read the data from the dataset"""

        f = os.path.join(self.predata_dir, 'MH_01_easy.p')
        if True and os.path.exists(f):
            return

        print("Start read_data, be patient please")
        def set_path(seq):
            path_imu = os.path.join(data_dir, seq, "mav0", "imu0", "data.csv")
            path_gt = os.path.join(data_dir, seq, "mav0", "state_groundtruth_estimate0", "data.csv")
            return path_imu, path_gt

        sequences = os.listdir(data_dir)
        # read each sequence
        for sequence in sequences:
            print("\nSequence name: " + sequence)
            path_imu, path_gt = set_path(sequence)
            imu = np.genfromtxt(path_imu, delimiter=",", skip_header=1)
            gt = np.genfromtxt(path_gt, delimiter=",", skip_header=1)

            # time synchronization between IMU and ground truth
            t0 = np.max([gt[0, 0], imu[0, 0]])
            t_end = np.min([gt[-1, 0], imu[-1, 0]])

            # start index
            idx0_imu = np.searchsorted(imu[:, 0], t0)
            idx0_gt = np.searchsorted(gt[:, 0], t0)

            # end index
            idx_end_imu = np.searchsorted(imu[:, 0], t_end, 'right')
            idx_end_gt = np.searchsorted(gt[:, 0], t_end, 'right')

            # subsample
            imu = imu[idx0_imu: idx_end_imu]
            gt = gt[idx0_gt: idx_end_gt]
            ts = imu[:, 0]/1e9

            # interpolate
            gt = self.interpolate(gt, gt[:, 0]/1e9, ts)

            # take ground truth position
            p_gt = gt[:, 1:4]
            p_gt = p_gt - p_gt[0]

            # take ground true quaternion pose
            q_gt = torch.Tensor(gt[:, 4:8]).double()
            q_gt = q_gt / q_gt.norm(dim=1, keepdim=True)
            Rot_gt = SO3.from_quaternion(q_gt, ordering='wxyz')

            # convert from numpy
            p_gt = torch.Tensor(p_gt).double()
            v_gt = torch.tensor(gt[:, 8:11]).double()
            imu = torch.Tensor(imu[:, 1:]).double()

            # compute pre-integration factors for all training
            mtf = self.min_train_freq
            dRot_ij = bmtm(Rot_gt[:-mtf], Rot_gt[mtf:])
            dRot_ij = SO3.dnormalize(dRot_ij)
            dxi_ij = SO3.log(dRot_ij).cpu()

            # save for all training
            mondict = {
                'xs': dxi_ij.float(),
                'us': imu.float(),
            }
            pdump(mondict, self.predata_dir, sequence + ".p")
            # save ground truth
            mondict = {
                'ts': ts,
                'qs': q_gt.float(),
                'vs': v_gt.float(),
                'ps': p_gt.float(),
            }
            pdump(mondict, self.predata_dir, sequence + "_gt.p")


class TUMVIDataset(BaseDataset):
    """
        Dataloader for the TUM-VI Data Set.
    """

    def __init__(self, data_dir, predata_dir, train_seqs, val_seqs,
                test_seqs, mode, N, min_train_freq, max_train_freq, dt=0.005,
                train_windows_per_seq=16):
        super().__init__(
            predata_dir,
            train_seqs,
            val_seqs,
            test_seqs,
            mode,
            N,
            min_train_freq,
            max_train_freq,
            dt,
            train_windows_per_seq,
        )
        # convert raw data to pre loaded data
        self.read_data(data_dir)
        self.finalize_preprocessing()
        # noise density
        self.imu_std = torch.Tensor([8e-5, 1e-3]).float()
        # bias repeatability (without in-run bias stability)
        self.imu_b0 = torch.Tensor([1e-3, 1e-3]).float()

    def read_data(self, data_dir):
        r"""Read the data from the dataset"""

        f = os.path.join(self.predata_dir, 'dataset-room1_512_16_gt.p')
        if True and os.path.exists(f):
            return

        print("Start read_data, be patient please")
        def set_path(seq):
            path_imu = os.path.join(data_dir, seq, seq, "mav0", "imu0", "data.csv")
            path_gt = os.path.join(data_dir, seq, seq, "mav0", "mocap0", "data.csv")
            return path_imu, path_gt

        sequences = os.listdir(data_dir)

        # read each sequence
        for sequence in sequences:
            print("\nSequence name: " + sequence)
            if 'room' not in sequence:
                continue

            path_imu, path_gt = set_path(sequence)
            imu = np.genfromtxt(path_imu, delimiter=",", skip_header=1)
            gt = np.genfromtxt(path_gt, delimiter=",", skip_header=1)

            # time synchronization between IMU and ground truth
            t0 = np.max([gt[0, 0], imu[0, 0]])
            t_end = np.min([gt[-1, 0], imu[-1, 0]])

            # start index
            idx0_imu = np.searchsorted(imu[:, 0], t0)
            idx0_gt = np.searchsorted(gt[:, 0], t0)

            # end index
            idx_end_imu = np.searchsorted(imu[:, 0], t_end, 'right')
            idx_end_gt = np.searchsorted(gt[:, 0], t_end, 'right')

            # subsample
            imu = imu[idx0_imu: idx_end_imu]
            gt = gt[idx0_gt: idx_end_gt]
            ts = imu[:, 0]/1e9

            # interpolate
            t_gt = gt[:, 0]/1e9
            gt = self.interpolate(gt, t_gt, ts)

            # take ground truth position
            p_gt = gt[:, 1:4]
            p_gt = p_gt - p_gt[0]

            # take ground true quaternion pose
            q_gt = SO3.qnorm(torch.Tensor(gt[:, 4:8]).double())
            Rot_gt = SO3.from_quaternion(q_gt, ordering='wxyz')

            # convert from numpy
            p_gt = torch.Tensor(p_gt).double()
            v_gt = torch.zeros_like(p_gt).double()
            v_gt[1:] = (p_gt[1:]-p_gt[:-1])/self.dt
            imu = torch.Tensor(imu[:, 1:]).double()

            # compute pre-integration factors for all training
            mtf = self.min_train_freq
            dRot_ij = bmtm(Rot_gt[:-mtf], Rot_gt[mtf:])
            dRot_ij = SO3.dnormalize(dRot_ij)
            dxi_ij = SO3.log(dRot_ij).cpu()

            # masks with 1 when ground truth is available, 0 otherwise
            masks = dxi_ij.new_ones(dxi_ij.shape[0])
            tmp = np.searchsorted(t_gt, ts[:-mtf])
            diff_t = ts[:-mtf] - t_gt[tmp]
            masks[np.abs(diff_t) > 0.01] = 0

            # save all the sequence
            mondict = {
                'xs': torch.cat((dxi_ij, masks.unsqueeze(1)), 1).float(),
                'us': imu.float(),
            }
            pdump(mondict, self.predata_dir, sequence + ".p")

            # save ground truth
            mondict = {
                'ts': ts,
                'qs': q_gt.float(),
                'vs': v_gt.float(),
                'ps': p_gt.float(),
            }
            pdump(mondict, self.predata_dir, sequence + "_gt.p")


class UZHFPVDataset(BaseDataset):
    """
        Dataloader for the UZH FPV dataset text/ZIP exports.
    """

    def __init__(self, data_dir, predata_dir, train_seqs, val_seqs,
                test_seqs, mode, N, min_train_freq, max_train_freq, dt=0.002,
                train_windows_per_seq=16):
        super().__init__(
            predata_dir,
            train_seqs,
            val_seqs,
            test_seqs,
            mode,
            N,
            min_train_freq,
            max_train_freq,
            dt,
            train_windows_per_seq,
        )
        self.read_data(data_dir)
        self.finalize_preprocessing()
        self.imu_std = torch.Tensor([8e-5, 1e-3]).float()
        self.imu_b0 = torch.Tensor([1e-3, 1e-3]).float()

    def read_data(self, data_dir):
        missing = [
            seq for seq in self.all_requested_sequences
            if not os.path.exists(os.path.join(self.predata_dir, seq + ".p"))
        ]
        if not missing:
            return

        print("Start read_data, be patient please")
        for sequence in self.all_requested_sequences:
            if os.path.exists(os.path.join(self.predata_dir, sequence + ".p")):
                continue
            print("\nSequence name: " + sequence)
            seq_dir = os.path.join(data_dir, sequence + "_snapdragon_with_gt")
            path_imu = os.path.join(seq_dir, "imu.txt")
            path_gt = os.path.join(seq_dir, "groundtruth.txt")
            if not os.path.isfile(path_imu) or not os.path.isfile(path_gt):
                raise FileNotFoundError(
                    f"Missing UZH FPV files for {sequence!r}: {path_imu!r}, {path_gt!r}"
                )

            imu_raw = np.genfromtxt(path_imu, comments="#", dtype=np.float64)
            gt_raw = np.genfromtxt(path_gt, comments="#", dtype=np.float64)
            imu_raw = _ensure_2d(imu_raw)
            gt_raw = _ensure_2d(gt_raw)
            imu_raw = imu_raw[np.isfinite(imu_raw[:, 1])]
            gt_raw = gt_raw[np.isfinite(gt_raw[:, 0])]

            # IMU text: index, timestamp_s, gyro_xyz, accel_xyz.
            imu_t = imu_raw[:, 1]
            imu_data = imu_raw[:, 2:8]

            # GT text: timestamp_s, position_xyz, quaternion_xyzw.
            gt = np.zeros((gt_raw.shape[0], 11), dtype=np.float64)
            gt[:, 0] = gt_raw[:, 0]
            gt[:, 1:4] = gt_raw[:, 1:4]
            gt[:, 4:8] = gt_raw[:, [7, 4, 5, 6]]
            gt[:, 8:11] = np.gradient(gt[:, 1:4], gt[:, 0], axis=0)

            t0 = np.max([gt[0, 0], imu_t[0]])
            t_end = np.min([gt[-1, 0], imu_t[-1]])
            idx0_imu = np.searchsorted(imu_t, t0)
            idx0_gt = np.searchsorted(gt[:, 0], t0)
            idx_end_imu = np.searchsorted(imu_t, t_end, "right")
            idx_end_gt = np.searchsorted(gt[:, 0], t_end, "right")

            imu_t = imu_t[idx0_imu:idx_end_imu]
            imu_data = imu_data[idx0_imu:idx_end_imu]
            gt = gt[idx0_gt:idx_end_gt]
            gt = self.interpolate(gt, gt[:, 0], imu_t)
            ts = imu_t

            p_gt = gt[:, 1:4]
            p_gt = p_gt - p_gt[0]
            q_gt = SO3.qnorm(torch.Tensor(gt[:, 4:8]).double())
            Rot_gt = SO3.from_quaternion(q_gt, ordering="wxyz")
            v_gt = torch.tensor(gt[:, 8:11]).double()
            imu = torch.Tensor(imu_data).double()

            mtf = self.min_train_freq
            dRot_ij = bmtm(Rot_gt[:-mtf], Rot_gt[mtf:])
            dRot_ij = SO3.dnormalize(dRot_ij)
            dxi_ij = SO3.log(dRot_ij).cpu()

            mondict = {
                "xs": dxi_ij.float(),
                "us": imu.float(),
            }
            pdump(mondict, self.predata_dir, sequence + ".p")
            mondict = {
                "ts": ts,
                "qs": q_gt.float(),
                "vs": v_gt.float(),
                "ps": torch.Tensor(p_gt).float(),
            }
            pdump(mondict, self.predata_dir, sequence + "_gt.p")


class BLACKBIRDDataset(BaseDataset):
    """
    Dataloader for the Blackbird UAV dataset.

    Expected official per-flight layout under `data_dir`:
        <trajectory>/<yawMode>/<speed>/
            rosbag.bag
            groundTruthPoses.csv
            flightNormalizationOffset.csv
            csv/
                blackbird_slash_imu.csv
                blackbird_slash_state.csv
                ...

    Integration notes from the official Blackbird repo:
    - IMU CSV is exported from topic `/blackbird/imu`
    - State CSV is exported from topic `/blackbird/state`
    - State pose is published for `body_frame`, while IMU lives in a fixed,
      rotated `imu` frame. We rotate body-frame quaternions into the IMU frame
      before forming orientation increments.
    """

    def __init__(self, data_dir, predata_dir, train_seqs, val_seqs,
                 test_seqs, mode, N, min_train_freq, max_train_freq, dt=0.01,
                 train_windows_per_seq=16):
        super().__init__(
            predata_dir,
            train_seqs,
            val_seqs,
            test_seqs,
            mode,
            N,
            min_train_freq,
            max_train_freq,
            dt,
            train_windows_per_seq,
        )
        self.read_data(data_dir)
        self.finalize_preprocessing()

    @staticmethod
    def _sequence_root(data_dir, sequence):
        return os.path.join(data_dir, *sequence.split('/'))

    @staticmethod
    def _load_imu(imu_path):
        imu = _load_blackbird_csv(imu_path, BLACKBIRD_IMU_USECOLS)
        imu[:, 0] = _to_seconds(imu[:, 0])
        return imu

    @staticmethod
    def _load_state(state_path):
        state = _load_blackbird_csv(state_path, BLACKBIRD_STATE_USECOLS)
        state[:, 0] = _to_seconds(state[:, 0])
        # bagToCsv preserves PoseStamped field order: position xyz then orientation xyzw.
        return np.column_stack((state[:, 0], state[:, 1:4], state[:, 7], state[:, 4:7]))

    @staticmethod
    def _load_groundtruth_fallback(gt_path):
        candidates = []
        for skip_header in (1, 0):
            try:
                gt = np.genfromtxt(gt_path, delimiter=",", skip_header=skip_header, dtype=np.float64)
            except ValueError:
                continue
            gt = _ensure_2d(gt)
            if gt.shape[1] < 8:
                continue
            gt = gt[np.isfinite(gt[:, 0])]
            if gt.size == 0:
                continue
            candidates.append(gt)
        if not candidates:
            raise ValueError(f"Could not parse Blackbird ground-truth CSV at {gt_path!r}")
        gt = candidates[0]
        gt[:, 0] = _to_seconds(gt[:, 0])
        # Fallback assumption follows the official pose-offset script's 8-column pose list:
        # timestamp, position xyz, quaternion xyzw.
        return np.column_stack((gt[:, 0], gt[:, 1:4], gt[:, 7], gt[:, 4:7]))

    def read_data(self, data_dir):
        if not self.all_requested_sequences:
            return

        print("Start read_data, be patient please")
        q_body_to_imu = BLACKBIRD_BODY_TO_IMU_QUAT_WXYZ.view(1, 4)

        for sequence in self.all_requested_sequences:
            seq_cache = os.path.join(self.predata_dir, sequence + ".p")
            gt_cache = os.path.join(self.predata_dir, sequence + "_gt.p")
            if os.path.exists(seq_cache) and os.path.exists(gt_cache):
                continue

            seq_root = self._sequence_root(data_dir, sequence)
            imu_path = os.path.join(seq_root, "csv", "blackbird_slash_imu.csv")
            state_path = os.path.join(seq_root, "csv", "blackbird_slash_state.csv")
            gt_path = os.path.join(seq_root, "groundTruthPoses.csv")

            if not os.path.exists(imu_path):
                raise FileNotFoundError(
                    f"Missing Blackbird IMU CSV for sequence {sequence!r}: {imu_path!r}"
                )
            if not os.path.exists(state_path) and not os.path.exists(gt_path):
                raise FileNotFoundError(
                    f"Missing Blackbird GT CSV for sequence {sequence!r}: expected {state_path!r} "
                    f"or {gt_path!r}"
                )

            print("\nSequence name: " + sequence)
            imu = self._load_imu(imu_path)
            gt = self._load_state(state_path) if os.path.exists(state_path) else self._load_groundtruth_fallback(gt_path)

            t0 = np.max([gt[0, 0], imu[0, 0]])
            t_end = np.min([gt[-1, 0], imu[-1, 0]])

            idx0_imu = np.searchsorted(imu[:, 0], t0)
            idx0_gt = np.searchsorted(gt[:, 0], t0)

            idx_end_imu = np.searchsorted(imu[:, 0], t_end, 'right')
            idx_end_gt = np.searchsorted(gt[:, 0], t_end, 'right')

            imu = imu[idx0_imu: idx_end_imu]
            gt = gt[idx0_gt: idx_end_gt]
            ts = imu[:, 0]

            gt = self.interpolate(gt, gt[:, 0], ts)

            p_gt = gt[:, 1:4]
            p_gt = p_gt - p_gt[0]

            q_body = SO3.qnorm(torch.tensor(gt[:, 4:8]).double())
            q_gt = SO3.qnorm(SO3.qmul(q_body, q_body_to_imu.expand(q_body.shape[0], 4)))
            Rot_gt = SO3.from_quaternion(q_gt, ordering='wxyz')

            p_gt = torch.tensor(p_gt).double()
            v_gt = torch.zeros_like(p_gt).double()
            v_gt[1:] = (p_gt[1:] - p_gt[:-1]) / self.dt
            imu = torch.tensor(imu[:, 1:]).double()

            mtf = self.min_train_freq
            dRot_ij = bmtm(Rot_gt[:-mtf], Rot_gt[mtf:])
            dRot_ij = SO3.dnormalize(dRot_ij)
            dxi_ij = SO3.log(dRot_ij).cpu()

            mondict = {
                'xs': dxi_ij.float(),
                'us': imu.float(),
            }
            pdump(mondict, self.predata_dir, sequence + ".p")

            mondict = {
                'ts': ts,
                'qs': q_gt.float(),
                'vs': v_gt.float(),
                'ps': p_gt.float(),
            }
            pdump(mondict, self.predata_dir, sequence + "_gt.p")


class KITTiDataset(BaseDataset):
    """
    Dataloader for the KITTI Data Set.
    """

    def __init__(self, data_dir, predata_dir, train_seqs, val_seqs,
                 test_seqs, mode, N, min_train_freq, max_train_freq, dt=0.005,
                 train_windows_per_seq=16):
        super().__init__(
            predata_dir,
            train_seqs,
            val_seqs,
            test_seqs,
            mode,
            N,
            min_train_freq,
            max_train_freq,
            dt,
            train_windows_per_seq,
        )
        # Convert raw data to pre-loaded data
        self.read_data(data_dir)
        self.finalize_preprocessing()

    def read_data(self, data_dir):
        """Read the data from the dataset"""

        f = os.path.join(self.predata_dir, '2011_09_26_drive_0001_sync_gt.p')
        if os.path.exists(f):
            return

        print("Start read_data, be patient please")

        def set_path(seq):
            path_imu = os.path.join(data_dir, seq, seq[0:10],seq,"oxts", "data.csv")
            path_gt = os.path.join(data_dir, seq, seq[0:10],seq, "oxts", "data.csv")
            return path_imu, path_gt

        sequences = os.listdir(data_dir)

        # Read each sequence
        for sequence in sequences:
            print("\nSequence name: " + sequence)
            path_imu, path_gt = set_path(sequence)
            imu = np.genfromtxt(path_imu, delimiter=",", skip_header=1)
            gt = np.genfromtxt(path_gt, delimiter=",", skip_header=1)

            # Time synchronization between IMU and ground truth
            t0 = np.max([gt[0, 0], imu[0, 0]])
            t_end = np.min([gt[-1, 0], imu[-1, 0]])

            # Start index
            idx0_imu = np.searchsorted(imu[:, 0], t0)
            idx0_gt = np.searchsorted(gt[:, 0], t0)

            # End index
            idx_end_imu = np.searchsorted(imu[:, 0], t_end, 'right')
            idx_end_gt = np.searchsorted(gt[:, 0], t_end, 'right')

            # Subsample
            imu = imu[idx0_imu: idx_end_imu]
            gt = gt[idx0_gt: idx_end_gt]
            ts = imu[:, 0] / 1e9

            # Interpolate
            t_gt = gt[:, 0] / 1e9
            gt = self.interpolate(gt, t_gt, ts)

            # Take ground truth position
            p_gt = gt[:, 1:4]
            p_gt = p_gt - p_gt[0]

            # Take ground truth quaternion pose
            q_gt = SO3.qnorm(torch.Tensor(gt[:, 4:8]).double())
            Rot_gt = SO3.from_quaternion(q_gt, ordering='wxyz')

            # Convert from numpy
            p_gt = torch.Tensor(p_gt).double()
            v_gt = torch.zeros_like(p_gt).double()
            v_gt[1:] = (p_gt[1:] - p_gt[:-1]) / self.dt
            imu = torch.Tensor(imu[:, 1:]).double()

            # Compute pre-integration factors for all training
            mtf = self.min_train_freq
            dRot_ij = bmtm(Rot_gt[:-mtf], Rot_gt[mtf:])
            dRot_ij = SO3.dnormalize(dRot_ij)
            dxi_ij = SO3.log(dRot_ij).cpu()

            # Masks with 1 when ground truth is available, 0 otherwise
            masks = dxi_ij.new_ones(dxi_ij.shape[0])
            tmp = np.searchsorted(t_gt, ts[:-mtf])
            diff_t = ts[:-mtf] - t_gt[tmp]
            masks[np.abs(diff_t) > 0.01] = 0

            # Save all the sequence
            mondict = {
                'xs': torch.cat((dxi_ij, masks.unsqueeze(1)), 1).float(),
                'us': imu.float(),
            }
            self.pdump(mondict, self.predata_dir, sequence + ".p")

            # Save ground truth
            mondict = {
                'ts': ts,
                'qs': q_gt.float(),
                'vs': v_gt.float(),
                'ps': p_gt.float(),
            }
            self.pdump(mondict, self.predata_dir, sequence + "_gt.p")

    def pdump(self, obj, path, filename):
        with open(os.path.join(path, filename), 'wb') as f:
            pickle.dump(obj, f)
