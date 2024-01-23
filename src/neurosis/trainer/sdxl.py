import warnings

from neurosis.trainer.cli import main

if __name__ == "__main__":
    warnings.warn("This module is deprecated, use `neurosis.trainer.cli` instead.", DeprecationWarning)
    main()
