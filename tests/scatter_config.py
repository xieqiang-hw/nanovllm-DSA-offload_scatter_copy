"""Branch-specific payload and runtime defaults for the unified test tools."""

DTYPE = 'c8'
BYTES_PER_TOKEN = 656
COPY_CAP_DEFAULT = 16384
DEFAULT_VISIBLE_DEVICES = '0,1,2,3,4,5,6,7'
LEGACY_VISIBLE_ENV = 'A5_TEST_VISIBLE_DEVICES'
RESULTS_SUBDIR = "results/scatter_copy_a5_c8"
MULTIPROCESS_TEST = 'a5_kvcache_scatter_copy_c8_multiprocess'
