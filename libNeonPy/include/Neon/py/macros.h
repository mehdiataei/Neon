#include <iostream>

#pragma once

#if 0
#define NEON_PY_PRINT_BEGIN(handel) \
    std::cout <<"NEON BEGIN "<<__func__<<  " (void* handle = " << handel << ")"<<std::endl;

 #define NEON_PY_PRINT_END(handel) \
    std::cout <<"NEON   END "<<__func__<<  " (void* handle = " << handel << ")"<<std::endl;
#endif

#define NEON_PY_PRINT_BEGIN(handel)
#define NEON_PY_PRINT_END(handel)