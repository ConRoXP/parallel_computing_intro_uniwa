/* ΕΙΣΑΓΩΓΗ ΣΤΟΝ ΠΑΡΑΛΛΗΛΟ ΥΠΟΛΟΓΙΣΜΟ - ΕΡΓΑΣΤΗΡΙΟ */
/* ΡΟΜΠΕΡΤ ΠΟΛΟΒΙΝΑ 23390338 - Εξάμηνο 7ο */
/* ΤΜΗΜΑ Ε5 - ΑΣΚΗΣΗ 2 */

// Μεταγλώττιση (OpenMPI): mpicc -o ask2 ask2_E5_23390338.c
// Εκτέλεση: mpiexec -n 4 ./ask2

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv){
    int proc_rank, total_ranks;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &total_ranks); //total processes (ranks)
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);  //rank of process making the call
    _Bool safe_free= 0;

    //Variables used by all ranks
    int input, N, rows, local_elements;
    double *A, *B, *C, *D, *R1, *R2, *R4;
    double *local_A, *local_B, *local_C, *local_D, *local_R1, *local_R2;
    double *crow, *drow, *rrow;
    double R3= 0;

    while(1){
        //Ι/Ο only on rank 0
        if(proc_rank== 0){
            printf("\n=== Menu ===\n");
            printf("1 to enter data - 2 to exit\n");
            scanf("%d", &input);

            if(input== 1){
                printf("Enter size N of square arrays\n(For questions 1, 2, 3: N mod p=0 - For q. 4: N=p)\n");
                scanf("%d", &N);

                A= (double*)malloc(N*sizeof(double));
                B= (double*)malloc(N*sizeof(double));
                R2= (double*)malloc(N*sizeof(double));
                //Allocating "2D" arrays with a single pointer using
                //pointer arithmetic
                //Ideal way for handling MPI_Scatter calls compared
                //to allocating with double pointers
                C= (double*)malloc((N*N)*sizeof(double));
                D= (double*)malloc((N*N)*sizeof(double));
                R1= (double*)malloc((N*N)*sizeof(double));
                R4= (double*)malloc((N*N)*sizeof(double));

                printf("\nEnter elements for array A\n");
                for(int i= 0; i< N; i++){
                    printf("[%d]: ", i);
                    fflush(stdout);
                    scanf("%lf", &A[i]);
                }

                printf("\nEnter elements for array B\n");
                for(int i= 0; i< N; i++){
                    printf("[%d]: ", i);
                    fflush(stdout);
                    scanf("%lf", &B[i]);
                }

                printf("\nEnter elements for array C\n");
                for(int i= 0; i< N; i++){
                    for(int j= 0; j< N; j++){
                        printf("[%d,%d]: ", i, j);
                        fflush(stdout);
                        scanf("%lf", &C[i*N+j]);
                    }
                }
                
                printf("\nEnter elements for array D\n");
                for(int i= 0; i< N; i++){
                    for(int j= 0; j< N; j++){
                        printf("[%d,%d]: ", i, j);
                        fflush(stdout);
                        scanf("%lf", &D[i*N+j]);
                    }
                }
            }
        }

        //Broadcasting menu choice
        MPI_Bcast(&input, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if(input== 1) safe_free= 1;
        else if(input== 2) break;
        else{
            if(proc_rank== 0) printf("Invalid choice\n");
            continue;
        }

        //Bcasting size Ν
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
        rows= N/total_ranks;
        local_elements= rows* N;

        //Allocating 1D arrays
        //Question 1
        local_C= (double*)malloc(local_elements*sizeof(double));
        local_D= (double*)malloc(local_elements*sizeof(double));
        local_R1= (double*)malloc(local_elements*sizeof(double));
        //Q. 2
        if(proc_rank> 0) B= (double*)malloc(N*sizeof(double));
        local_R2= (double*)malloc(N*sizeof(double));
        //Q. 3
        local_A= (double*)malloc(rows*sizeof(double));
        local_B= (double*)malloc(rows*sizeof(double));
        //Q. 4
        crow= (double*)malloc(N*sizeof(double));
        drow= (double*)malloc(N*sizeof(double));
        rrow= (double*)calloc(N, sizeof(double));

        //Q.1 - Assigning workload to each rank
        //Each rank gets assigned N/total_ranks rows
        MPI_Scatter(C, local_elements, MPI_DOUBLE, local_C, local_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(D, local_elements, MPI_DOUBLE, local_D, local_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        //Q.1 - Calculating sum and gathering results to r0
        for(int k= 0; k< local_elements; k++){
            local_R1[k]= local_C[k]+ local_D[k];
        }
        MPI_Gather(local_R1, local_elements, MPI_DOUBLE, R1, local_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        //Bcasting Β for Q.2
        MPI_Bcast(B, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        //Q.2 - Multiplying arrays and gathering results to r0
        for(int i= 0; i< rows; i++){
            double sum= 0;
            for(int j= 0; j< N; j++){
                sum+= local_C[i*N+j]*B[j];
            }
            local_R2[i]= sum;
        }
        MPI_Gather(local_R2, rows, MPI_DOUBLE, R2, rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        //Q.3 - Scattering arrays Α και Β
        //Each rank gets (rows= N/total_ranks) elements from each array
        MPI_Scatter(A, rows, MPI_DOUBLE, local_A, rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(B, rows, MPI_DOUBLE, local_B, rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        //Q.3 - Performing multiplication and gathering results to r0
        double local_sum= 0;
        for(int i= 0; i< rows; i++){
            local_sum+= local_A[i]*local_B[i];
        }
        MPI_Reduce(&local_sum, &R3, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        //Q.4
        //Each rank gets 1 row of each array (C and D)
        MPI_Scatter(C, N, MPI_DOUBLE, crow, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(D, N, MPI_DOUBLE, drow, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        int prev= (proc_rank-1 + total_ranks) % total_ranks; //**
        int next= (proc_rank+ 1) % total_ranks; //**
        
        int id= proc_rank;
        //operation: line C * line D
        //via pointer arithmetics we're basically doing:
        //crow[id] == C[proc_rank][id]
        //drow[k] == D[id][k]
        for(int step= 0; step< total_ranks; step++){
            for(int k= 0; k< N; k++){
                rrow[k]+= crow[id]* drow[k];
            }
            //In each step we send the current drow to the previous processor-neighbor
            //and we receive the next from the next processor-neighbor
            MPI_Send(drow, N, MPI_DOUBLE, prev, 10, MPI_COMM_WORLD);
            MPI_Recv(drow, N, MPI_DOUBLE, next, 10, MPI_COMM_WORLD, &status);

            id= (id+1) % total_ranks; //moving crow index to the next element
            //** we use mod total_ranks because otherwise in edge cases (rank= 0 || rank= total_ranks-1),
            //we'll go out of bounds.
        }
        MPI_Gather(rrow, N, MPI_DOUBLE, R4, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        //Rank 0 printing results
        if(proc_rank== 0){
            printf("\nI. R1= C(%dx%d)+D(%dx%d)=\n", N, N, N, N);
            for(int i= 0; i< N; i++){
                for(int j= 0; j< N; j++)
                    printf("%.2f  ", R1[i*N+j]);
                printf("\n");
            }
            printf("\n");
            printf("II. R2= C(%dx%d)+B(%dx1)=\n", N, N, N);
            for(int i= 0; i< N; i++){
                printf("%.1f\n", R2[i]);
            }
            printf("\n");
            printf("III. R3= A(1x%d)*B(%dx1)= %1.f\n", N, N, R3);
            printf("\n");
            printf("IV. R4= C(%dx%d)*D(%dx%d)=\n", N, N, N, N);
            for(int i= 0; i< N; i++){
                for(int j= 0; j< N; j++)
                    printf("%.2f  ", R4[i*N+j]);
                printf("\n");
            }
        }
    }

    if(safe_free== 1){
        free(B); free(local_A); free(local_B);
        free(local_C); free(local_D);
        free(local_R1); free(local_R2);
        free(crow); free(drow); free(rrow);

        if(proc_rank== 0){
            free(A); free(C); free(D);
            free(R1); free(R2); free(R4);
        }
    }
    MPI_Finalize();
    return 0;
}
