# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# Runs one MOAB example built against MOAB (REF) and against soma::compat
# (SOMA) in fresh directories under WORK, with the same arguments (ARGS, '|'
# for ';'), after copying INPUTS ('|'-separated) into both. Their standard
# output must be equal, and each file named in OUTPUTS ('|'-separated) must
# read the same in MOAB (DESCRIBE compares them). PART_FILE, when set, writes a
# partition file of that many lines (line i holds (7 i) mod 4) into both
# directories first.

foreach(v REF SOMA WORK DESCRIBE)
  if(NOT DEFINED ${v})
    message(FATAL_ERROR "compare_example: ${v} is not set")
  endif()
endforeach()
string(REPLACE "|" ";" args "${ARGS}")
string(REPLACE "|" ";" inputs "${INPUTS}")
string(REPLACE "|" ";" outputs "${OUTPUTS}")

foreach(side ref soma)
  file(REMOVE_RECURSE "${WORK}/${side}")
  file(MAKE_DIRECTORY "${WORK}/${side}")
  foreach(f IN LISTS inputs)
    file(COPY "${f}" DESTINATION "${WORK}/${side}")
  endforeach()
  if(PART_FILE)
    set(text "")
    math(EXPR last "${PART_FILE} - 1")
    foreach(i RANGE ${last})
      math(EXPR p "(7 * ${i}) % 4")
      string(APPEND text "${p}\n")
    endforeach()
    file(WRITE "${WORK}/${side}/part.txt" "${text}")
  endif()
endforeach()

execute_process(COMMAND "${REF}" ${args} WORKING_DIRECTORY "${WORK}/ref"
  OUTPUT_VARIABLE ref_out ERROR_VARIABLE ref_err RESULT_VARIABLE ref_rc)
execute_process(COMMAND "${SOMA}" ${args} WORKING_DIRECTORY "${WORK}/soma"
  OUTPUT_VARIABLE soma_out ERROR_VARIABLE soma_err RESULT_VARIABLE soma_rc)
if(NOT ref_rc EQUAL 0)
  message(FATAL_ERROR "the MOAB build failed (${ref_rc}):\n${ref_out}\n${ref_err}")
endif()
if(NOT soma_rc EQUAL 0)
  message(FATAL_ERROR "the soma::compat build failed (${soma_rc}):\n${soma_out}\n${soma_err}")
endif()
if(NOT ref_out STREQUAL soma_out)
  file(WRITE "${WORK}/ref.stdout" "${ref_out}")
  file(WRITE "${WORK}/soma.stdout" "${soma_out}")
  message(FATAL_ERROR "standard output differs: compare ${WORK}/ref.stdout and ${WORK}/soma.stdout")
endif()

foreach(f IN LISTS outputs)
  foreach(side ref soma)
    if(NOT EXISTS "${WORK}/${side}/${f}")
      message(FATAL_ERROR "the ${side} run wrote no ${f}")
    endif()
  endforeach()
  execute_process(COMMAND "${DESCRIBE}" "${WORK}/ref/${f}" "${WORK}/soma/${f}"
    OUTPUT_VARIABLE diff ERROR_VARIABLE err RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "${f} reads differently in MOAB:\n${diff}${err}")
  endif()
endforeach()
message(STATUS "same output${outputs}")
