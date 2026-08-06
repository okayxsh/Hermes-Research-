# Runtime forecasting

The autopilot derives its forecast only from measured real pilot durations. It reports optimistic, median, conservative, serial, approved-parallel, GPU-hour, disk/log, and timeout-bound estimates. Timeouts are upper bounds, not expected durations. Acquisition always has one worker; evaluation defaults to one and can use two only after approved benchmark evidence proves at least 30% throughput improvement without correctness, isolation, or resource regressions.
