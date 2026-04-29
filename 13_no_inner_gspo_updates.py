import sys

from run_final_experiments import main


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--experiments",
        "13_no_inner_gspo_updates",
        *sys.argv[1:],
    ]
    main()
