try:
    from .six_state_engine.cli import main
except ImportError:
    from six_state_engine.cli import main

if __name__ == "__main__":
    main()
