#include <iostream>

#pragma once

#if 1
#define NEON_PY_PRINT_BEGIN(handel) \
    std::cout <<"NEON BEGIN "<<__func__<<  " (void* handle = " << handel << ")"<<std::endl;

 #define NEON_PY_PRINT_END(handel) \
    std::cout <<"NEON   END "<<__func__<<  " (void* handle = " << handel << ")"<<std::endl;
#else
#define NEON_PY_PRINT_BEGIN(handel)
#define NEON_PY_PRINT_END(handel)
#endif
