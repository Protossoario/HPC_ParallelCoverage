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
load_locations (char *fn, double locs[][2], int first)
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
coverage (double locs[][2], int nlocs, double *result, int *npaths, int *ncoin)
{
    int np_overall = 0, nc = 0, np, i, j, block_size, block_end;
    double hmsum = 0.0, dsum, hm, d, y1, x1, y2, x2, global_hmsum;

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
        printf("    #%d: hmsum = %f\n", world_rank, hmsum);
    }

    printf("  Local hmsum #%d = %f\n", world_rank, hmsum);

    MPI_Reduce(&hmsum, &global_hmsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // add up all the local np_overall's and nc's to compute the global sum total for each variable
    // then pass the result into the output variable parameters
    MPI_Reduce(&np_overall, npaths, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&nc, ncoin, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("Global hmsum = %f\n", global_hmsum);
        printf("Nlocs = %d\n", nlocs);
        // compute the coverage result based on the global sum total of hmsum
        *result = nlocs / global_hmsum;
        printf("Result = %f\n", *result);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char fn[MAXFN];
    double locs[MAXLOCS][2];
    double cov;
    int nlocs, np, nc;
    sprintf (fn, "/home/alien/data/%s_%s%d.txt", "ebr", "bark", 1);
    nlocs = load_locations (fn, locs, 0);
    coverage (locs, nlocs, &cov, &np, &nc);

    if (world_rank == 0) {
        printf ("  %s %f %d %d\n", fn, cov, np, nc);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
