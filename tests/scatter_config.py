"""Branch-specific payload and runtime defaults for the unified test tools."""

DTYPE = 'bf16'
BYTES_PER_TOKEN = 1152
COPY_CAP_DEFAULT = 2048
DEFAULT_VISIBLE_DEVICES = '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15'
LEGACY_VISIBLE_ENV = 'A3_TEST_VISIBLE_DEVICES'
RESULTS_SUBDIR = "results/scatter_copy_a3"
MULTIPROCESS_TEST = 'kvcache_scatter_copy_multiprocess'
