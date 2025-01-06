#include "Neon/Neon.h"
#include "Neon/Report.h"
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
    Neon::init();
    auto reportPtr = new (std::nothrow) Neon::Report(name);
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

    auto report = reinterpret_cast<Neon::Report*>(*handle);

    if (report != nullptr) {
        delete report;
    }
    *handle = nullptr;
    NEON_PY_PRINT_END(*handle);
    return 0;
}

extern "C" auto report_add_member_string(void*       handle,
                       const char* memberKey,
                       const char*     memberVal) -> void
{
    auto report = reinterpret_cast<Neon::core::Report*>(handle);
    report->addMember(memberKey, memberVal);
}

extern "C" auto report_add_member_int64(void*       handle,
                       const char* memberKey,
                       const int64_t     memberVal) -> void
{
    auto report = reinterpret_cast<Neon::core::Report*>(handle);
    report->addMember(memberKey, memberVal);
}


extern "C" auto report_add_member_vector_int64(void*       handle,
                              const char* memberKey,
                              int         vec_len,
                              const int64_t*    vec_val) -> void
{
    auto           report = reinterpret_cast<Neon::core::Report*>(handle);
// initialize an std::vector from the row pointer

    std::vector<int64_t> vec(vec_val, vec_val+vec_len);
    report->addMember(memberKey, vec);
}

extern "C" auto report_add_member_double(void*       handle,
                       const char* memberKey,
                       const double     memberVal) -> void
{
    auto report = reinterpret_cast<Neon::core::Report*>(handle);
    report->addMember(memberKey, memberVal);
}


extern "C" auto report_add_member_vector_double(void*       handle,
                              const char* memberKey,
                              int           vec_len,
                              const double*    vec_val) -> void
{
    auto           report = reinterpret_cast<Neon::core::Report*>(handle);
    // initialize an std::vector from the row pointer

    std::vector<double> vec(vec_val, vec_val+vec_len);
    report->addMember(memberKey, vec);
}

extern "C" auto report_write(void*       handle,
                             const char* fname,
                             bool        append_time_to_file) -> void
{
    auto report = reinterpret_cast<Neon::core::Report*>(handle);
    report->write(fname, append_time_to_file);
}