from experiments.benchmarks.cutest_runner import (
    _configure_pycutest_linux_link_libraries,
)


def test_pycutest_links_glibc_vector_math_when_available():
    class InstallScripts:
        setupScriptLinux = "libraries=['gfortran']\n"

    scripts = InstallScripts()
    _configure_pycutest_linux_link_libraries(
        scripts, platform="linux", find_library=lambda name: "libmvec.so.1"
    )
    assert scripts.setupScriptLinux == "libraries=['gfortran','mvec']\n"

    _configure_pycutest_linux_link_libraries(
        scripts, platform="linux", find_library=lambda name: "libmvec.so.1"
    )
    assert scripts.setupScriptLinux.count("mvec") == 1


def test_pycutest_does_not_require_vector_math_when_unavailable():
    class InstallScripts:
        setupScriptLinux = "libraries=['gfortran']\n"

    scripts = InstallScripts()
    _configure_pycutest_linux_link_libraries(
        scripts, platform="linux", find_library=lambda name: None
    )
    assert scripts.setupScriptLinux == "libraries=['gfortran']\n"
