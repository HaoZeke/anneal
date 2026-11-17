A meson build with `with_eigen=true` configures again when neither pkg-config nor CMake finds `eigen3`.
The lookup passed `dependency()` a `subproject:` keyword, which current meson rejects; it now names the fallback `['eigen', 'eigen_dep']`, so Eigen comes from `subprojects/eigen.wrap`.
