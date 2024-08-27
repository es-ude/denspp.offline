#ifndef usenrnmpi_h
#define usenrnmpi_h

/* define to 1 if you want MPI specific features activated */
#define NRNMPI 1

/* define to 1 if you want parallel distributed cells (and gap junctions) */
#define PARANEURON 1

/* define to 1 if you want mpi dynamically loaded instead of linked normally */
#define NRNMPI_DYNAMICLOAD 1

/* define to 1 if you want the MUSIC - MUlti SImulation Coordinator */
/* #undef NRN_MUSIC */

/* define to the dll path if you want to load automatically */
#define DLL_DEFAULT_FNAME "x86_64/.libs/libnrnmech.so"

/* define if needed */
/* #undef ALWAYS_CALL_MPI_INIT */

/* Number of times to retry a failed open */
/* #undef FILE_OPEN_RETRY */

/* Define bits for BGPDMA & 1 (ISend) & 2 (DMA spike transfer) & 4 (DMA Record Replay */
#define BGPDMA 1

/* Define to 1 for possibility of rank 0 xopen/ropen a file and broadcast everywhere */
/* #undef USE_NRNFILEWRAP */

#endif
