import sys

from run_final_experiments import main


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--experiments",
        "09_refresh_only_negative_pending",
        *sys.argv[1:],
    ]
    main()
