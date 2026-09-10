"""Branch-specific payload and runtime defaults for the unified test tools."""

DTYPE = 'bf16'
BYTES_PER_TOKEN = 1152
COPY_CAP_DEFAULT = 2048
DEFAULT_VISIBLE_DEVICES = '0,1,2,3,4,5,6,7'
LEGACY_VISIBLE_ENV = 'A5_TEST_VISIBLE_DEVICES'
RESULTS_SUBDIR = "results/scatter_copy_a5"
MULTIPROCESS_TEST = 'a5_kvcache_scatter_copy_multiprocess'
