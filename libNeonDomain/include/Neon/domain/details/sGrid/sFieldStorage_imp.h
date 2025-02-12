#pragma once
#include "Neon/domain/details/sGrid/sFieldStorage.h"

namespace Neon::domain::details::sGrid {

template <typename OuterGridT, typename T, int C>
sFieldStorage<OuterGridT, T, C>::sFieldStorage()
{
}

template <typename OuterGridT, typename T, int C>
sFieldStorage<OuterGridT, T, C>::sFieldStorage(const Neon::domain::interface::GridBase& gb)
{
    for (const auto& exec : ExecutionUtils::getAllOptions()) {
        for (const auto& dw : DataViewUtil::validOptions()) {
            partitions[Neon::ExecutionUtils::toInt(exec)][Neon::DataViewUtil::toInt(dw)] = gb.getDevSet().template newDataSet<Partition>();
        }
    }
}

template <typename OuterGridT, typename T, int C>
auto sFieldStorage<OuterGridT, T, C>::getPartition(Neon::Execution execution,
                                                   Neon::DataView  dataView,
                                                   Neon::SetIdx    setIdx)
    -> Partition&
{
    return partitions[Neon::ExecutionUtils::toInt(execution)][Neon::DataViewUtil::toInt(dataView)][setIdx];
}

template <typename OuterGridT, typename T, int C>
auto sFieldStorage<OuterGridT, T, C>::getPartition(Neon::Execution execution,
                                                   Neon::DataView  dataView,
                                                   Neon::SetIdx    setIdx) const -> const Partition&
{
    return partitions[Neon::ExecutionUtils::toInt(execution)][Neon::DataViewUtil::toInt(dataView)][setIdx];
}


template <typename OuterGridT, typename T, int C>
auto sFieldStorage<OuterGridT, T, C>::getPartitionSet(Neon::Execution execution, Neon::DataView dataView) -> Neon::set::DataSet<Partition>&
{
    return partitions[Neon::ExecutionUtils::toInt(execution)][Neon::DataViewUtil::toInt(dataView)];
}

}  // namespace Neon::domain::details::sGrid
