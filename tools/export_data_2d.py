import argparse
import os.path as osp

from tools.dataset_converters import nuscenes_converter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_path", type=str, help="root path")
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="v1.0-trainval",
        help="data version",
    )
    parser.add_argument("--t4label", action="store_true", help="whether use T4label")
    parser.add_argument(
        "--no-mono3d",
        action="store_false",
        help="whether not to use mono3d",
    )
    args = parser.parse_args()

    for info_name in ("nuscenes_infos_train.pkl", "nuscenes_infos_val.pkl"):
        info_path = osp.join(args.root_path, info_name)
        nuscenes_converter.export_2d_annotation(
            args.root_path,
            info_path,
            args.version,
            mono3d=args.no_mono3d,
            use_t4label=args.t4label,
        )


if __name__ == "__main__":
    main()
