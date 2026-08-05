# Failure policy

Freeze this policy before final evaluation. Malformed calls get one repair attempt; timeouts get one retry; bridge/Ollama crashes restart the episode under a new attempt ID; invalid actions count toward the limit; interruptions restart from the beginning; partial attempts are never merged.
