import sys

from run_final_experiments import main


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--experiments",
        "04_always_keep_controller",
        *sys.argv[1:],
    ]
    main()
