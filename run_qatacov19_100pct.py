from run_qatacov19_benchmark import build_parser, run_ratio_experiment


def main() -> None:
    parser = build_parser(default_save_root="./runs/qatacov19_100pct")
    args = parser.parse_args()
    run_ratio_experiment(args, ratio=1.0)


if __name__ == "__main__":
    main()
