import sys

from run_final_experiments import main


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--experiments",
        "06_no_controller_regret_update",
        *sys.argv[1:],
    ]
    main()
