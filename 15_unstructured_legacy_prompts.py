import sys

from run_final_experiments import main


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--experiments",
        "15_unstructured_legacy_prompts",
        *sys.argv[1:],
    ]
    main()
