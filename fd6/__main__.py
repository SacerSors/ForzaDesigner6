import multiprocessing
import sys

if __name__ == "__main__":
    # Required for ProcessPoolExecutor under PyInstaller --onefile on Windows.
    # MUST run before any project imports — otherwise worker subprocesses
    # re-enter this module before the worker dispatch hook has been installed,
    # which can trigger ImportError("attempted relative import with no known
    # parent package") inside PyInstaller's frozen bootstrap. Imports stay
    # inside the guard so the worker subprocesses don't replay them either.
    multiprocessing.freeze_support()

    # Early PyTorch import on Linux to prevent static TLS exhaustion and LLVM symbol collisions
    # before GUI libraries or Mesa are loaded.
    if sys.platform.startswith("linux"):
        try:
            import torch  # noqa: F401
            try:
                multiprocessing.set_start_method("spawn")
            except RuntimeError:
                pass  # Start method already set
        except ImportError:
            pass

    from fd6.app import main
    raise SystemExit(main())
