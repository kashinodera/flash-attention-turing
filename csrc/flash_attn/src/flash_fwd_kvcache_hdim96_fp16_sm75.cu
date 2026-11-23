#include "flash_fwd_kvcache_launch.h"

template <>
void run_mha_fwd_kvcache_<96>(Flash_fwd_params& params) {
    run_mha_fwd_kvcache_hdim<96>(params);
}
