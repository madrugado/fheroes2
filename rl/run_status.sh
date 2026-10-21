#!/bin/bash
# Status of a joint training run on the training machine (rl/run_status.py over ssh).
# The address is NOT stored in the repo: set FHEROES2_SERVER (ssh arguments, e.g. "user@host -p PORT") and
# FHEROES2_DIR (the repo on that machine).
#   rl/run_status.sh joint16 ggv2 game_grpo_v2      # training log prefix, collector log prefix, data prefix
set -euo pipefail
: "${FHEROES2_SERVER:?set FHEROES2_SERVER to the ssh arguments of the training machine}"
: "${FHEROES2_DIR:?set FHEROES2_DIR to the repo path on the training machine}"
run=${1:?training run, e.g. joint16}
collect=${2:-}
data=${3:-}
# shellcheck disable=SC2086
ssh $FHEROES2_SERVER "cd $FHEROES2_DIR && . ./rl_env.sh && python rl/run_status.py --train logs/${run}_train.log \
    ${collect:+--collect 'logs/${collect}_c*.log'} ${data:+--data 'rl/data/${data}_c*.jsonl'} --mem logs/memwatch.log"
