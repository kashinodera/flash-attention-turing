#include "flash_fwd_kvcache_launch.h"

template <>
void run_mha_fwd_kvcache_<32>(Flash_fwd_params& params) {
    run_mha_fwd_kvcache_hdim<32>(params);
}
