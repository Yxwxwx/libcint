if(I8 OR PYPZPX)
  message(FATAL_ERROR "GPW requires 32-bit FINT and PySCF (px,py,pz) ordering")
endif()
add_library(gpw SHARED gpw_int2e.c gpw_int1e.c gpw_rys.c ../cart2sph.c ../fblas.c)
target_include_directories(gpw PUBLIC ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${PROJECT_BINARY_DIR}/include ${PROJECT_BINARY_DIR}/src ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(gpw PRIVATE m)
set_target_properties(gpw PROPERTIES C_VISIBILITY_PRESET hidden
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR} VERSION 0.2.0 SOVERSION 0)
install(TARGETS gpw LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${PROJECT_SOURCE_DIR}/include/gpw.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(ENABLE_TEST)
  enable_testing()
  add_executable(gpw_smoke ../../testsuite/gpw/smoke.c)
  target_include_directories(gpw_smoke PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_compile_options(gpw_smoke PRIVATE -UNDEBUG)
  target_link_libraries(gpw_smoke PRIVATE gpw m)
  add_test(NAME gpw_smoke COMMAND gpw_smoke)
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy, pyscf, mpmath"
    RESULT_VARIABLE gpw_python_status OUTPUT_QUIET ERROR_QUIET)
  if(gpw_python_status EQUAL 0)
    add_test(NAME gpw_int1e COMMAND ${PYTHON_EXECUTABLE}
      ${PROJECT_SOURCE_DIR}/testsuite/gpw/test_int1e.py --library $<TARGET_FILE:gpw>
      --report ${PROJECT_BINARY_DIR}/gpw_int1e_validation.json)
    set_tests_properties(gpw_int1e PROPERTIES ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
  else()
    message(STATUS "GPW Python numerical test requires numpy, pyscf and mpmath; C smoke is available")
  endif()
endif()
