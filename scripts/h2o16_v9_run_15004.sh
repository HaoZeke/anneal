#!/bin/bash
# Copyright (c) anneal developers
# wrapper header
# Every timed command for both arms:
#   sudo unshare --uts sh -c 'hostname bench; exec <cmd>'
set -u
exec sudo -n unshare --uts sh -c 'hostname bench; exec bash /home/ubuntu/work/launch-15004-bench.sh'
