#pragma once

#include "Neon/set/container/ContainerAPI.h"
#include "Neon/set/container/Loader.h"
#include "Neon/set/container/OldDeviceManagedContainer.h"

namespace Neon::set::internal {

/**
 * Specialized implementation of KContainer_i
 *
 *
 * @tparam t_DataIteratorContainer
 * @tparam t_ManagedLaunch
 */
struct DeviceThenHostManagedContainer : ContainerAPI
{
   public:
    virtual ~DeviceThenHostManagedContainer() override = default;

   private:
    std::shared_ptr<Neon::set::internal::ContainerAPI> mDevice;
    std::shared_ptr<Neon::set::internal::ContainerAPI> mHost;

   public:
    /**
     * User facing API to define a kernel
     * @param data
     * @param userLambda
     */
    DeviceThenHostManagedContainer(const std::string&                                  name,
                                   std::shared_ptr<Neon::set::internal::ContainerAPI> device,
                                   std::shared_ptr<Neon::set::internal::ContainerAPI> host)
    {
        mDevice = device;
        mHost = host;

        this->setDataViewSupport(DataViewSupport::off);

        setContainerExecutionType(ContainerExecutionType::deviceThenHostManaged);
        setName(name);
    }

    auto newLoader(Neon::Execution execution,
                   Neon::SetIdx     setIdx,
                   Neon::DataView   dataView,
                   LoadingMode_e::e loadingMode) -> Loader
    {
        auto loader = Loader(*this,
                             execution,
                             setIdx,
                             dataView,
                             loadingMode);
        return loader;
    }

    auto newParser() -> Loader
    {
        auto parser = Loader(*this,
                             Neon::Execution::host,
                             Neon::SetIdx(0),
                             Neon::DataView::STANDARD,
                             Neon::set::internal::LoadingMode_e::PARSE_AND_EXTRACT_LAMBDA);
        return parser;
    }

    auto parse() -> const std::vector<Neon::set::dataDependency::Token>& override
    {
        mHost->parse();
        mDevice->parse();

        auto const& hostTokens = mHost->getTokens();
        auto const& devTokens = mDevice->getTokens();

        for (auto const& token : devTokens) {
            getTokensRef().push_back(token);
        }
        std::vector<Neon::set::dataDependency::Token> filtered;
        for (auto const& token : hostTokens) {
            bool foundMatch = false;
            for (auto& acceptedTokens : getTokensRef()) {
                if (token.uid() == acceptedTokens.uid()) {
                    acceptedTokens.mergeAccess(token.access());
                    foundMatch = true;
                }
            }
            if (!foundMatch) {
                filtered.push_back(token);
            }
        }

        for (auto const& token : filtered) {
            getTokensRef().push_back(token);
        }

        return getTokens();
    }

    /**
     * Run container over streams
     * @param streamIdx
     * @param dataView
     */
    virtual auto run(int streamIdx = 0, Neon::DataView dataView = Neon::DataView::STANDARD) -> void override
    {
        if (Neon::DataView::STANDARD == dataView) {
            mDevice->run(streamIdx, Neon::DataView::STANDARD);
            mHost->run(streamIdx, Neon::DataView::STANDARD);
            return ;
        }
        NEON_THROW_UNSUPPORTED_OPTION("A DeviceThenHostManagedContainer object can not be run directly.");
    }

    virtual auto run(Neon::SetIdx   setIdx,
                     int            streamIdx = 0,
                     Neon::DataView dataView = Neon::DataView::STANDARD) -> void override
    {
        if (Neon::DataView::STANDARD == dataView) {
            mDevice->run(setIdx, streamIdx, Neon::DataView::STANDARD);
            mHost->run(setIdx, streamIdx, Neon::DataView::STANDARD);
        }
        NEON_THROW_UNSUPPORTED_OPTION("A DeviceThenHostManagedContainer object can not be run directly.");
    }

    auto getHostContainer() -> std::shared_ptr<ContainerAPI> final
    {
        return mHost;
    }

    auto getDeviceContainer() -> std::shared_ptr<ContainerAPI> final
    {
        return mDevice;
    }
};

}  // namespace Neon
