import sys

from run_final_experiments import main


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--experiments",
        "07_vote_threshold_state_machine",
        *sys.argv[1:],
    ]
    main()
