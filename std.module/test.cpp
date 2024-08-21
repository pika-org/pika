#include "allstd.hpp" // this is required to work by [std.modules]/4
#include <version>
import std;

#ifdef __GLIBCXX__
static constexpr auto Lib = "libstdc++";
#elif defined (_MSVC_STL_UPDATE)
static constexpr auto Lib = "ms-stl";
#else
static constexpr auto Lib = "libc++";
#endif
#if defined (_MSC_VER)
static constexpr auto Compiler = "MSVC";
static constexpr auto Major = _MSC_VER;
#else
static constexpr auto Compiler = "Clang";
static constexpr auto Major = __clang_major__;
#endif

auto g() -> std::generator<std::string> {
    co_yield Compiler;
    co_yield "import std;";
};


int main() {
    std::expected<int, bool> exp = 42;
    std::vector<std::string> vs;
    for (auto && s : g()) {
        vs.push_back(std::move(s));
    }
    std::println("{} {} & {} welcome '{}' !", vs[0], Major, Lib, vs[1]);
    return vs.empty() + *exp;
}