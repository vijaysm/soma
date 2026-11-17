# Python bindings: the extension module _soma (nanobind) inside the package soma.
# The build tree is importable with PYTHONPATH=<build>/python.

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE soma_nanobind_dir OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE soma_nanobind_found)
if(NOT soma_nanobind_found EQUAL 0)
  message(FATAL_ERROR "SOMA_ENABLE_PYTHON: nanobind is not installed for ${Python_EXECUTABLE}")
endif()
find_package(nanobind CONFIG REQUIRED PATHS "${soma_nanobind_dir}" NO_DEFAULT_PATH)

set(soma_python_dir "${CMAKE_BINARY_DIR}/python/soma")
nanobind_add_module(_soma NB_STATIC
  src/module.cpp src/runtime.cpp src/mesh.cpp src/edit.cpp src/fields.cpp src/io.cpp src/parallel.cpp)
target_link_libraries(_soma PRIVATE soma::soma)
# The library keeps fields in std::any and compares types across translation
# units; with nanobind's hidden visibility the binding sources would see other
# type_info objects for the same template instances (macOS arm64), and every
# any_cast across the boundary would fail. The module keeps default visibility.
set_target_properties(_soma PROPERTIES CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF)
set_target_properties(_soma PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${soma_python_dir}")
if(SOMA_ENABLE_MPI)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import mpi4py; print(mpi4py.get_include())"
    OUTPUT_VARIABLE soma_mpi4py_include OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE soma_mpi4py_found)
  if(NOT soma_mpi4py_found EQUAL 0)
    message(FATAL_ERROR "SOMA_ENABLE_PYTHON with MPI needs mpi4py for ${Python_EXECUTABLE}")
  endif()
  target_include_directories(_soma PRIVATE "${soma_mpi4py_include}")
endif()

# The package's Python files, copied next to the extension at build time.
file(GLOB soma_python_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/soma/*.py")
set(soma_python_copies)
foreach(src IN LISTS soma_python_sources)
  get_filename_component(name "${src}" NAME)
  add_custom_command(OUTPUT "${soma_python_dir}/${name}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${src}" "${soma_python_dir}/${name}"
    DEPENDS "${src}")
  list(APPEND soma_python_copies "${soma_python_dir}/${name}")
endforeach()
add_custom_target(soma_python_files ALL DEPENDS ${soma_python_copies})
add_dependencies(_soma soma_python_files)

if(SOMA_INSTALL)
  if(NOT SOMA_PYTHON_INSTALL_DIR)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('platlib'))"
      OUTPUT_VARIABLE soma_platlib OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(SOMA_PYTHON_INSTALL_DIR "${soma_platlib}" CACHE PATH "Directory the Python package soma is installed in")
  endif()
  install(TARGETS _soma LIBRARY DESTINATION "${SOMA_PYTHON_INSTALL_DIR}/soma")
  install(FILES ${soma_python_sources} DESTINATION "${SOMA_PYTHON_INSTALL_DIR}/soma")
endif()

# soma_add_python_test(<file> RANKS ...): pytest on <file>, once per rank count under
# mpiexec (once, serially, without MPI).
function(soma_add_python_test file)
  cmake_parse_arguments(T "" "" "RANKS" ${ARGN})
  get_filename_component(stem "${file}" NAME_WE)
  set(run "${Python_EXECUTABLE}" -m pytest -x -q -p no:cacheprovider "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
  set(env "PYTHONPATH=${CMAKE_BINARY_DIR}/python" "OMP_NUM_THREADS=1" "PYTHONDONTWRITEBYTECODE=1")
  if(SOMA_ENABLE_MPI)
    foreach(np IN LISTS T_RANKS)
      add_test(NAME python.${stem}.np${np}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} ${run} ${MPIEXEC_POSTFLAGS})
      set_tests_properties(python.${stem}.np${np} PROPERTIES TIMEOUT 60 PROCESSORS ${np} ENVIRONMENT "${env}")
    endforeach()
  else()
    add_test(NAME python.${stem} COMMAND ${run})
    set_tests_properties(python.${stem} PROPERTIES TIMEOUT 60 ENVIRONMENT "${env}")
  endif()
endfunction()

if(SOMA_ENABLE_TESTS)
  soma_add_python_test(tests/test_runtime.py RANKS 1 2 3 4)
  soma_add_python_test(tests/test_mesh.py RANKS 1 2 3 4)
  soma_add_python_test(tests/test_edit.py RANKS 1)
  soma_add_python_test(tests/test_fields.py RANKS 1 2 3 4)
  soma_add_python_test(tests/test_parallel.py RANKS 1 2 3 4)
  if(SOMA_ENABLE_HDF5)
    soma_add_python_test(tests/test_io.py RANKS 1 2 3 4)
  endif()
endif()
