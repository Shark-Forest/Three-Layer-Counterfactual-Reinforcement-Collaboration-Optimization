import sys

from run_final_experiments import main


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--experiments",
        "05_no_counterfactual_controller_values",
        *sys.argv[1:],
    ]
    main()
