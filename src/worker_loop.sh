#!/bin/bash
# Keep one worker alive on a device until every job has an output. Usage: worker_loop.sh <device> <tag>
dev=$1; tag=$2
while true; do
  python -u src/run_escape.py results/jobs.json "$dev" >> logs/worker_$tag.log 2>&1
  rc=$?
  n=$(python -c "import json,os;print(sum(not os.path.exists(f\"results/escape/{j['id']}.npz\") for j in json.load(open('results/jobs.json'))))")
  echo "$(date +%T) rc=$rc remaining=$n" >> logs/worker_$tag.log
  [ "$n" = "0" ] && break
  sleep 30
done
