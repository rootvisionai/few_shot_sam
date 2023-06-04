#!/bin/bash

# Start the first process
python ./backend/server.py -D
status=$?
if [ $status -ne 0 ]; then
  echo "Failed to start ./backend/server.py: $status"
  exit $status
fi

# Start the second process
python ./backend/forwarder.py -D
status=$?
if [ $status -ne 0 ]; then
  echo "Failed to start ./backend/forwarder.py: $status"
  exit $status
fi

# Naive check runs checks once a minute to see if either of the processes exited.
# This is not a robust mechanism for process supervision, but it can serve as a simple example.

while sleep 60; do
  ps aux |grep server.py |grep -q -v grep
  PROCESS_1_STATUS=$?
  ps aux |grep forwarder.py |grep -q -v grep
  PROCESS_2_STATUS=$?
  # If either process has exited, exit.
  if [ $PROCESS_1_STATUS -ne 0 -o $PROCESS_2_STATUS -ne 0 ]; then
    echo "One of the processes has already exited."
    exit 1
  fi
done
