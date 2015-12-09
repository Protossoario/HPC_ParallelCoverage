//-----------------------------------------------------------------------------
// c c o v e r a g e   --   calculate the coverage of feature detectors
//-----------------------------------------------------------------------------
// COMPILE: gcc -Wall -g -o ccoverage ccoverage.c -lm
//-----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define MAXFN   200
#define MAXLINE 2000
#define MAXLOCS 200000

int world_size;
int world_rank;

int
load_locations (char *fn, float locs[][2], int first)
{
    char line[MAXLINE], *kwd;
    int nlocs;
    FILE *f;

    if ((f = fopen (fn, "r")) == NULL) {
        fprintf (stderr, "Cannot open %s!\n", fn);
        exit (EXIT_FAILURE);
    }
    fgets (line, MAXLINE, f);
    fgets (line, MAXLINE, f);
    nlocs = first;
    while (fgets (line, MAXLINE, f) != NULL) {
        kwd = strtok (line, " ");
        locs[nlocs][1] = atof (kwd);
        kwd = strtok ((char *) NULL, " ");
        locs[nlocs][0] = atof (kwd);
        nlocs += 1;
        if (nlocs >= MAXLOCS) {
            fprintf (stderr, "Not enough space to hold all locations: %d!\n", nlocs);
            exit (EXIT_FAILURE);
        }
    }
    fclose (f);
    return nlocs;
}


void
coverage (float locs[][2], int nlocs, float *result, int *npaths, int *ncoin)
{
    int np_overall = 0, nc = 0, np, i, j, block_size, block_end;
    float hmsum = 0.0, dsum, hm, d, y1, x1, y2, x2, global_hmsum;

    // ceiling integer division of nlocs / world_size
    // obtains the block size for each processor to calculate a partial sum of coverage
    block_size = ((nlocs - 1) / world_size) + 1;

    // initial index is given by block size * processor index (or rank)
    i = world_rank * block_size;

    // stop index is given by the start of the next processor's block
    block_end = world_rank * block_size + block_size;

    while (i < block_end && i < nlocs) {
        y1 = locs[i][0];
        x1 = locs[i][1];
        np = 0;
        dsum = 0.0;
        for (j = 0; j < nlocs; j++) {
            if (i != j) {
                y2 = locs[j][0] - y1;
                x2 = locs[j][1] - x1;
                d = sqrt (y2 * y2 + x2 * x2);
                if (d <= 0.0) {
                    nc += 1;
                } else {
                    dsum += 1.0 / d;
                    np += 1;
                    np_overall += 1;
                }
            }
        }
        hm = np / dsum;
        hmsum += 1.0 / hm;
        i++;
    }

    MPI_Reduce(&hmsum, &global_hmsum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    // add up all the local np_overall's and nc's to compute the global sum total for each variable
    // then pass the result into the output variable parameters
    MPI_Reduce(&np_overall, npaths, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&nc, ncoin, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // wait for sync between all processors before continuing
    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) {
        // compute the coverage result based on the global sum total of hmsum
        *result = nlocs / global_hmsum;
    }
}


float
mean_coverage1 (char *opname, char **imsets, int nimsets)
{
    int nfiles = 0, i, fno, nlocs, np, nc;
    float locs[MAXLOCS][2];
    float csum = 0.0, cov;
    char fn[MAXFN];

    for (i = 0; i < nimsets; i++) {
        for (fno = 1; fno < 7; fno++) {
            sprintf (fn, "/home/alien/data/%s_%s%d.txt", opname, imsets[i], fno);
            nlocs = load_locations (fn, locs, 0);
            coverage (locs, nlocs, &cov, &np, &nc);
            // printf ("  %s %s %f %d %d\n", opname, fn, cov, np, nc);
            if (world_rank == 0) {
                csum += cov;
                nfiles += 1;
            }
        }
    }
    if (world_rank == 0) {
        return csum / nfiles;
    } else {
        return 0;
    }
}


float
mean_coverage2 (char *opname1, char *opname2, char **imsets, int nimsets)
{
    int nfiles = 0, i, fno, nlocs, np, nc;
    float locs[MAXLOCS][2];
    float csum = 0.0, cov;
    char fn1[MAXFN], fn2[MAXFN];

    for (i = 0; i < nimsets; i++) {
        for (fno = 1; fno < 7; fno++) {
            sprintf (fn1, "/home/alien/data/%s_%s%d.txt", opname1, imsets[i], fno);
            nlocs = load_locations (fn1, locs, 0);
            sprintf (fn2, "/home/alien/data/%s_%s%d.txt", opname2, imsets[i], fno);
            nlocs = load_locations (fn2, locs, nlocs);
            coverage (locs, nlocs, &cov, &np, &nc);
            // printf ("  %s %s %f %d %d\n", fn1, fn2, cov, np, nc);
            if (world_rank == 0) {
                csum += cov;
                nfiles += 1;
            }
        }
    }
    if (world_rank == 0) {
        return csum / nfiles;
    } else {
        return 0;
    }
}


float
mean_coverage3 (char *opname1, char *opname2, char *opname3,
                char **imsets, int nimsets)
{
    int nfiles = 0, i, fno, nlocs, np, nc;
    float locs[MAXLOCS][2];
    float csum = 0.0, cov;
    char fn1[MAXFN], fn2[MAXFN], fn3[MAXFN];

    for (i = 0; i < nimsets; i++) {
        for (fno = 1; fno < 7; fno++) {
            sprintf (fn1, "/home/alien/data/%s_%s%d.txt", opname1, imsets[i], fno);
            nlocs = load_locations (fn1, locs, 0);
            sprintf (fn2, "/home/alien/data/%s_%s%d.txt", opname2, imsets[i], fno);
            nlocs = load_locations (fn2, locs, nlocs);
            sprintf (fn3, "/home/alien/data/%s_%s%d.txt", opname3, imsets[i], fno);
            nlocs = load_locations (fn3, locs, nlocs);
            coverage (locs, nlocs, &cov, &np, &nc);
            // printf ("  %s %s %s %f %d %d\n", fn1, fn2, fn3, cov, np, nc);
            if (world_rank == 0) {
                csum += cov;
                nfiles += 1;
            }
        }
    }
    if (world_rank == 0) {
        return csum / nfiles;
    } else {
        return 0;
    }
}


float
mean_coverage4 (char *opname1, char *opname2, char *opname3, char *opname4,
                char **imsets, int nimsets)
{
    int nfiles = 0, i, fno, nlocs, np, nc;
    float locs[MAXLOCS][2];
    float csum = 0.0, cov;
    char fn1[MAXFN], fn2[MAXFN], fn3[MAXFN], fn4[MAXFN];

    for (i = 0; i < nimsets; i++) {
        for (fno = 1; fno < 7; fno++) {
            sprintf (fn1, "/home/alien/data/%s_%s%d.txt", opname1, imsets[i], fno);
            nlocs = load_locations (fn1, locs, 0);
            sprintf (fn2, "/home/alien/data/%s_%s%d.txt", opname2, imsets[i], fno);
            nlocs = load_locations (fn2, locs, nlocs);
            sprintf (fn3, "/home/alien/data/%s_%s%d.txt", opname3, imsets[i], fno);
            nlocs = load_locations (fn3, locs, nlocs);
            sprintf (fn4, "/home/alien/data/%s_%s%d.txt", opname4, imsets[i], fno);
            nlocs = load_locations (fn4, locs, nlocs);
            coverage (locs, nlocs, &cov, &np, &nc);
            // printf ("  %s %s %s %s %f %d %d\n", fn1, fn2, fn3, fn4, cov, np, nc);
            if (world_rank == 0) {
                csum += cov;
                nfiles += 1;
            }
        }
    }
    if (world_rank == 0) {
        return csum / nfiles;
    } else {
        return 0;
    }
}


int
main (int argc, char **argv)
{
    char *opnames[] = {"ebr", "ibr", "mser", "sfop"};
    int nopnames = 4;
    char *imsets[] = {"bark", "bikes", "boat", "graf", "leuv", "trees", "ubc",
    "wall"};
    int nimsets = 8;
    int i1, i2, i3, i4;
    float mc;

    /*
    * MPI initialization
    */

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    /*
    * End of MPI initialization
    */

    // Calculate the coverage for each individual operator.
    for (i1 = 0; i1 < nopnames; i1++) {
        mc = mean_coverage1 (opnames[i1], imsets, nimsets);
        if (world_rank == 0) {
            printf ("%s: %f\n", opnames[i1], mc);
        }
    }

    // Calculate the coverage for each combination of pairs of operators.
    for (i1 = 0; i1 < nopnames; i1++) {
        for (i2 = i1+1; i2 < nopnames; i2++) {
            mc = mean_coverage2 (opnames[i1], opnames[i2], imsets, nimsets);
            if (world_rank == 0) {
                printf ("%s + %s: %f\n", opnames[i1], opnames[i2], mc);
            }
        }
    }

    // Calculate the coverage for each combination of triplets of operators.
    for (i1 = 0; i1 < nopnames; i1++) {
        for (i2 = i1+1; i2 < nopnames; i2++) {
            for (i3 = i2+1; i3 < nopnames; i3++) {
                mc = mean_coverage3 (opnames[i1], opnames[i2], opnames[i3],
                                     imsets, nimsets);
                if (world_rank == 0) {
                    printf ("%s + %s + %s: %f\n", opnames[i1], opnames[i2], opnames[i3], mc);
                }
            }
        }
    }

    // Calculate the coverage for each combination of quadruplets of operators.
    for (i1 = 0; i1 < nopnames; i1++) {
        for (i2 = i1+1; i2 < nopnames; i2++) {
            for (i3 = i2+1; i3 < nopnames; i3++) {
                for (i4 = i3+1; i4 < nopnames; i4++) {
                    mc = mean_coverage4 (opnames[i1], opnames[i2], opnames[i3],
                                         opnames[i4],  imsets, nimsets);
                    if (world_rank == 0) {
                        printf ("%s + %s + %s + %s: %f\n", opnames[i1], opnames[i2],
                                                           opnames[i3], opnames[i4], mc);
                    }
                }
            }
        }
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}

//-----------------------------------------------------------------------------
// End of ccoverage.c
//-----------------------------------------------------------------------------
