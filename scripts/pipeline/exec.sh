#!/bin/bash
# Run an arbitrary command in the machine's container (replaces scripts/incontainer.sh):  exec.sh --machine local3070 "python eval/predict.py ..."
source "$(dirname "$0")/lib.sh"; init "$@"
exec bash -c "${POSITIONAL[*]}"
