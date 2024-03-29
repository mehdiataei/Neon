#include "Neon/py/grid.h"
#include "Neon/domain/Grids.h"

auto grid_new(uint64_t& handle) -> int
{
    std::cout << "grid_new - BEGIN"<<std::endl;

    Neon::init();

    using Grid = Neon::dGrid;
    Neon::Backend         bk(1, Neon::Runtime::openmp);
    Neon::index_3d        dim(10, 10, 10);
    Neon::domain::Stencil d3q19 = Neon::domain::Stencil::s19_t(false);
    Grid g (bk, dim, [](auto idx) { return true; }, d3q19);
    auto                  gridPtr = new (std::nothrow) Grid(g);

    if (gridPtr == nullptr) {
        std::cout << "NeonPy: Initialization error. Unable to allocage grid " << std::endl;
        return -1;
    }
    handle = (uint64_t)gridPtr;
    std::cout << "grid_new - END"<<std::endl;

    // g.ioDomainToVtk("")
    return 0;
}

auto grid_delete(uint64_t& handle) -> int
{
    std::cout << "grid_delete - BEGIN"<<std::endl;

    using Grid = Neon::dGrid;
    Grid* gridPtr = (Grid*)handle;

    if (gridPtr != nullptr) {
        delete gridPtr;
    }
    std::cout << "grid_delete - END"<<std::endl;
    return 0;
}


auto field_new(uint64_t& handle, uint64_t& gridHandle) -> int
{
    std::cout << "field_new - BEGIN"<<std::endl;

    using Grid = Neon::dGrid;
    Grid* gridPtr = (Grid*)gridHandle;
    Grid& grid = *gridPtr;

    if (gridPtr != nullptr) {
        using Field = Grid::Field<int,1>;
        Field field= grid.newField<int,1>("test", 1, 0,Neon::DataUse::HOST_DEVICE);
        Field * fieldPtr = new (std::nothrow) Field(field  );
        if (fieldPtr == nullptr) {
            std::cout << "NeonPy: Initialization error. Unable to allocage grid " << std::endl;
            return -1;
        }
        handle = (uint64_t)fieldPtr;
        std::cout << "field_new - END"<<std::endl;

        return 0;
    }
    return -1;
}

auto field_delete(uint64_t& handle) -> int
{
    std::cout << "field_delete - BEGIN"<<std::endl;

    using Grid = Neon::dGrid;
    using Field = Grid::Field<int,1>;

    Field* fieldPtr = (Field*)handle;

    if (fieldPtr != nullptr) {
        delete fieldPtr;
    }
    std::cout << "field_delete - END"<<std::endl;

    return 0;
}