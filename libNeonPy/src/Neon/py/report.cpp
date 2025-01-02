#include "Neon/core/tools/Report.h"
#include "Neon/py/macros.h"

/**
 * Initialize a new grid object on the heap.
 * NOTE: some parameters are still not exposed
 */
extern "C" auto report_new(
    void**      handle,
    const char* name)
    -> int
{
    auto reportPtr = new (std::nothrow) Neon::core::Report(name);
    if (reportPtr != nullptr) {
        *handle = reportPtr;
        return 0;
    }
    return 1;
}

/**
 * Delete a grid object on the heap.
 */
extern "C" auto report_delete(
    void** handle)
    -> int
{
    NEON_PY_PRINT_BEGIN(*handle);

    auto report = reinterpret_cast<Neon::core::Report*>(*handle);

    if (report != nullptr) {
        delete report;
    }
    *handle = nullptr;
    NEON_PY_PRINT_END(*handle);
    return 0;
}

template <typename T>
auto report_add_member(void*       handle,
                       const char* memberKey,
                       const T     memberVal) -> void
{
    auto report = reinterpret_cast<Neon::core::Report*>(handle);
    report->addMember(memberKey, memberVal);
    return;
}

extern "C"
{
    template void report_add_member<uint32_t>(void* handle, const char*, const uint32_t);
    template void report_add_member<int32_t>(void* handle, const char*, const int32_t);
    template void report_add_member<uint64_t>(void* handle, const char*, const uint64_t);
    template void report_add_member<int64_t>(void* handle, const char*, const int64_t);
    template void report_add_member<const char*>(void* handle, const char*, const char*);
    template void report_add_member<double>(void* handle, const char*, const double);
    template void report_add_member<float>(void* handle, const char*, const float);
    template void report_add_member<bool>(void* handle, const char*, const bool);
    template void report_add_member<std::string>(void* handle, const char*, const std::string);
}

template <typename T>
auto report_add_member_vector(void*       handle,
                              const char* memberKey,
                              int         vec_len,
                              const T*    vec_val) -> void
{
    auto           report = reinterpret_cast<Neon::core::Report*>(handle);
    std::vector<T> vec(vec_val, vec_len);
    report->addMember(memberKey, vec);
    return;
}

extern "C"
{
    extern template void report_add_member_vector<uint32_t>(void* handle, const char* memberKey, int vec_len, uint32_t const*);
    extern template void report_add_member_vector<int32_t>(void* handle, const char* memberKey, int vec_len, int32_t const*);
    extern template void report_add_member_vector<double>(void* handle, const char* memberKey, int vec_len, double const*);
    extern template void report_add_member_vector<float>(void* handle, const char* memberKey, int vec_len, float const*);
    extern template void report_add_member_vector<bool>(void* handle, const char* memberKey, int vec_len, bool const*);
}

extern "C" auto report_write(void*       handle,
                             const char* fname,
                             bool        append_time_to_file) -> void
{
    auto report = reinterpret_cast<Neon::core::Report*>(handle);
    report->write(fname, append_time_to_file);
    return;
}