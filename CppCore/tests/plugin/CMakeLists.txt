add_library(rgpot_null SHARED ../../examples/plugins/null_plugin.c)
add_library(rgpot_morse SHARED ../../examples/plugins/morse_plugin.c)
add_library(rgpot_coulomb SHARED ../../examples/plugins/coulomb_plugin.f90
                                    ../../examples/plugins/coulomb_plugin_wrapper.c)
add_library(rgpot_plugin_lifetime SHARED ../plugin_lifetime_fixture.c)
foreach(_plugin rgpot_null rgpot_morse rgpot_coulomb rgpot_plugin_lifetime)
  target_include_directories(${_plugin} PRIVATE "${PROJECT_SOURCE_DIR}/include")
  set_target_properties(${_plugin} PROPERTIES PREFIX "lib")
endforeach()
if(UNIX)
  target_link_libraries(rgpot_morse PRIVATE m)
endif()
if(APPLE)
  # Link with the C driver so the wrapper's libSystem references resolve at
  # the C toolchain's deployment target, with the Fortran runtime added
  # explicitly.
  set_target_properties(rgpot_coulomb PROPERTIES LINKER_LANGUAGE C)
  target_link_directories(rgpot_coulomb PRIVATE
                          ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES})
  target_link_libraries(rgpot_coulomb PRIVATE
                        ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
endif()
foreach(_test PluginTest PluginContractTest ExternalPluginTest ExternalFortranPluginTest)
  rgpot_test_executable(${_test} ../${_test}.cc)
  rgpot_add_test(${_test} COMMAND ${_test}
    ENVIRONMENT "RGPOT_TEST_PLUGIN_DIR=$<TARGET_FILE_DIR:rgpot_null>")
endforeach()
rgpot_test_executable(PluginLifetimeTest ../PluginLifetimeTest.cc)
rgpot_add_test(PluginLifetimeTest
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/../test_plugin_lifetime.py"
          $<TARGET_FILE:PluginLifetimeTest> $<TARGET_FILE:rgpot_plugin_lifetime>)
