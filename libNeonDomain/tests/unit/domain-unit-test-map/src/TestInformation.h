#pragma once
namespace {
struct TestInformation
{
    static auto prefix() -> std::string
    {
        return "domain-unit-test-map";
    }

    static auto fullName(const std::string& gridName)
    {
        return prefix() + "-" + gridName;
    }
};
}  // namespace