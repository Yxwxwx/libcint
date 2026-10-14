if(I8 OR PYPZPX)
  message(FATAL_ERROR "GPW requires 32-bit FINT and PySCF (px,py,pz) ordering")
endif()
add_library(gpw SHARED gpw_int2e.c gpw_int1e.c gpw_rys.c ../cart2sph.c ../fblas.c
  ../rys_roots.c ../rys_wheeler.c ../fmt.c ../eigh.c ../find_roots.c)
target_include_directories(gpw PUBLIC ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${PROJECT_BINARY_DIR}/include ${PROJECT_BINARY_DIR}/src ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(gpw PRIVATE m)
option(GPW_WITH_OPENMP "Parallelize GPW wavevector batches with OpenMP" ON)
if(GPW_WITH_OPENMP)
  find_package(OpenMP QUIET COMPONENTS C)
  if(OpenMP_C_FOUND)
    target_link_libraries(gpw PRIVATE OpenMP::OpenMP_C)
  else()
    message(STATUS "OpenMP C unavailable; building serial GPW")
  endif()
endif()
if(OpenMP_C_FOUND AND GPW_WITH_OPENMP AND CMAKE_C_COMPILER_ID STREQUAL "GNU")
  include(CheckCSourceCompiles)
  include(CMakePushCheckState)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_LIBRARIES m)
  check_c_source_compiles("
    typedef double V __attribute__((vector_size(16)));
    extern V _ZGVbN2v_exp(V), _ZGVbN2v_cos(V), _ZGVbN2v_sin(V);
    int main(void) { V x={0.1,0.2};
      x=_ZGVbN2v_exp(x)+_ZGVbN2v_cos(x)+_ZGVbN2v_sin(x); return x[0]==0; }
    " GPW_HAVE_SIMD_MATH)
  cmake_pop_check_state()
  if(GPW_HAVE_SIMD_MATH)
    target_compile_definitions(gpw PRIVATE GPW_SIMD_MATH)
    set_source_files_properties(gpw_rys.c PROPERTIES COMPILE_FLAGS "-fno-math-errno")
  endif()
endif()
target_compile_definitions(gpw PRIVATE KEEP_GOING)
if(QUADMATH_FOUND)
  target_link_libraries(gpw PRIVATE quadmath)
endif()
if(WITH_POLYNOMIAL_FIT)
  target_sources(gpw PRIVATE ../polyfits.c ../sr_rys_polyfits.c)
endif()
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
    add_test(NAME gpw_dipole COMMAND ${PYTHON_EXECUTABLE}
      ${PROJECT_SOURCE_DIR}/testsuite/gpw/test_dipole.py --library $<TARGET_FILE:gpw>
      --report ${PROJECT_BINARY_DIR}/gpw_dipole_validation.json)
    set_tests_properties(gpw_dipole PROPERTIES ENVIRONMENT
      "PYTHONDONTWRITEBYTECODE=1;OMP_NUM_THREADS=4;OPENBLAS_NUM_THREADS=1")
  else()
    message(STATUS "GPW Python numerical test requires numpy, pyscf and mpmath; C smoke is available")
  endif()
endif()
