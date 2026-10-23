import os

# Always run tests against the deterministic mocks.
os.environ["USE_MOCK_FALLBACK"] = "true"
