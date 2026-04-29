import sys

from run_final_experiments import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--experiments", "08_no_vote_updates", *sys.argv[1:]]
    main()
