import argparse
import os
import sys

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.dataset import BLACKBIRD_BODY_TO_IMU_QUAT_WXYZ  # noqa: E402


IMU_HEADER = [
    "rosbagTimestamp",
    "header.seq",
    "header.stamp.secs",
    "header.stamp.nsecs",
    "header.frame_id",
    "orientation.x",
    "orientation.y",
    "orientation.z",
    "orientation.w",
    "orientation_covariance",
    "angular_velocity.x",
    "angular_velocity.y",
    "angular_velocity.z",
    "angular_velocity_covariance",
    "linear_acceleration.x",
    "linear_acceleration.y",
    "linear_acceleration.z",
    "linear_acceleration_covariance",
]

STATE_HEADER = [
    "rosbagTimestamp",
    "header.seq",
    "header.stamp.secs",
    "header.stamp.nsecs",
    "header.frame_id",
    "pose.position.x",
    "pose.position.y",
    "pose.position.z",
    "pose.orientation.x",
    "pose.orientation.y",
    "pose.orientation.z",
    "pose.orientation.w",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Import the public UZH Blackbird sample into this repo's CSV layout.")
    parser.add_argument(
        "--src-root",
        default="/tmp/learned_inertial_model_odometry/datasets/Blackbird/clover/yawForward/maxSpeed5p0",
        help="Path containing train/val/test data.hdf5 sample splits.",
    )
    parser.add_argument(
        "--dest-root",
        default=os.path.join(BASE_DIR, "data", "BLACKBIRD", "dataset", "clover", "yawForward", "maxSpeed5p0"),
        help="Destination root for the converted sample.",
    )
    return parser.parse_args()


def _split_timestamp(ts):
    secs = np.floor(ts).astype(np.int64)
    nsecs = np.round((ts - secs.astype(np.float64)) * 1e9).astype(np.int64)
    nsecs[nsecs == 1_000_000_000] = 999_999_999
    return secs, nsecs


def _body_quaternion_from_imu(q_imu_xyzw):
    q_body_to_imu_xyzw = np.array(
        [
            float(BLACKBIRD_BODY_TO_IMU_QUAT_WXYZ[1]),
            float(BLACKBIRD_BODY_TO_IMU_QUAT_WXYZ[2]),
            float(BLACKBIRD_BODY_TO_IMU_QUAT_WXYZ[3]),
            float(BLACKBIRD_BODY_TO_IMU_QUAT_WXYZ[0]),
        ]
    )
    rot_imu = Rotation.from_quat(q_imu_xyzw)
    rot_body_to_imu = Rotation.from_quat(q_body_to_imu_xyzw)
    rot_body = rot_imu * rot_body_to_imu.inv()
    return rot_body.as_quat()


def _write_csv(path, header, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, data, delimiter=",", header=",".join(header), comments="", fmt="%.12f")


def convert_split(src_split_dir, dest_split_dir):
    h5_path = os.path.join(src_split_dir, "data.hdf5")
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"Missing sample HDF5: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        ts = np.asarray(f["ts"])
        gyro = np.asarray(f["gyro_raw"])
        accel = np.asarray(f["accel_raw"])
        traj = np.asarray(f["traj_target"])

    secs, nsecs = _split_timestamp(ts)
    n = ts.shape[0]
    seq = np.arange(n, dtype=np.int64)
    zeros = np.zeros(n, dtype=np.float64)

    imu = np.column_stack(
        [
            ts,
            seq,
            secs,
            nsecs,
            zeros,
            zeros,
            zeros,
            zeros,
            np.ones(n, dtype=np.float64),
            zeros,
            gyro[:, 0],
            gyro[:, 1],
            gyro[:, 2],
            zeros,
            accel[:, 0],
            accel[:, 1],
            accel[:, 2],
            zeros,
        ]
    )

    q_imu_xyzw = traj[:, 3:7]
    q_body_xyzw = _body_quaternion_from_imu(q_imu_xyzw)
    state = np.column_stack(
        [
            ts,
            seq,
            secs,
            nsecs,
            zeros,
            traj[:, 0],
            traj[:, 1],
            traj[:, 2],
            q_body_xyzw[:, 0],
            q_body_xyzw[:, 1],
            q_body_xyzw[:, 2],
            q_body_xyzw[:, 3],
        ]
    )

    csv_dir = os.path.join(dest_split_dir, "csv")
    _write_csv(os.path.join(csv_dir, "blackbird_slash_imu.csv"), IMU_HEADER, imu)
    _write_csv(os.path.join(csv_dir, "blackbird_slash_state.csv"), STATE_HEADER, state)


def main():
    args = parse_args()
    src_root = os.path.abspath(args.src_root)
    dest_root = os.path.abspath(args.dest_root)

    for split in ("train", "val", "test"):
        src_split_dir = os.path.join(src_root, split)
        if not os.path.isdir(src_split_dir):
            raise FileNotFoundError(f"Missing sample split directory: {src_split_dir}")
        dest_split_dir = os.path.join(dest_root, split)
        convert_split(src_split_dir, dest_split_dir)
        print(f"imported {split}: {dest_split_dir}")


if __name__ == "__main__":
    main()
