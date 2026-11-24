#!/usr/bin/env bash
# Run on your computer: copies kfcollect.zip to the server and updates it in one step.
#   ./deploy.sh azureuser@<server>              (zip next to this script, or pass a path as 2nd argument)
#   SSH_OPTS="-i ~/.ssh/azure.pem" ./deploy.sh azureuser@<server>
set -euo pipefail
HOST="${1:?usage: ./deploy.sh user@host [path/to/kfcollect.zip]}"
ZIP="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/kfcollect.zip}"
[[ -f "$ZIP" ]] || { echo "Can't find $ZIP"; exit 1; }
# shellcheck disable=SC2086
scp ${SSH_OPTS:-} "$ZIP" "$HOST:kfcollect.zip"
# shellcheck disable=SC2086
ssh -t ${SSH_OPTS:-} "$HOST" 'set -e; rm -rf ~/kfcollect; python3 -m zipfile -e ~/kfcollect.zip ~ && chmod +x ~/kfcollect/*.sh && sudo ~/kfcollect/update.sh'
