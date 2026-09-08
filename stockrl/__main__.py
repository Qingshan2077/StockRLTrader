"""Allow ``python -m stockrl`` to invoke the local CLI."""

from stockrl.cli import main


raise SystemExit(main())
