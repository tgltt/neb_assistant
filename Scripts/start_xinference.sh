#!/bin/bash
xinference launch -n bge-m3 -t embedding -f pytorch -u bge-m3-test -s 7 --host 0.0.0.0 --port 9997