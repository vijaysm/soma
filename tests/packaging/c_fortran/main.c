/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2022-2026, Vijay Mahadevan. */

/* A C program on an installed SOMA: a box mesh on this rank, its cells counted. */

#include <stdio.h>

#include "soma/c/soma.h"

int main(int argc, char** argv) {
#if SOMA_C_HAVE_MPI
  MPI_Init(&argc, &argv);
#else
  (void)argc;
  (void)argv;
#endif
  int rc = 1;
  soma_comm_t* self = NULL;
  soma_mesh_t* m = NULL;
  soma_lo_t cells = 0;
  if (soma_initialize() == SOMA_OK && soma_comm_self(&self) == SOMA_OK &&
      soma_box_mesh(self, 2, 2, 2, SOMA_HEX, 1.0, 1.0, 1.0, SOMA_ORDERING_FILE, &m) == SOMA_OK &&
      soma_mesh_count(m, 3, &cells) == SOMA_OK && cells == 8) {
    printf("c consumer: %ld cells\n", (long)cells);
    rc = 0;
  } else {
    char msg[256];
    soma_last_error(msg, sizeof msg);
    fprintf(stderr, "c consumer failed: %s\n", msg);
  }
  soma_mesh_free(m);
  soma_comm_free(self);
  soma_finalize();
#if SOMA_C_HAVE_MPI
  MPI_Finalize();
#endif
  return rc;
}
