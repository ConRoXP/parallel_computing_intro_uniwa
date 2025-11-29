#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv){
    int proc_rank, total_ranks;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &total_ranks); //total processes (ranks)
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);  //rank of current process

    //Variables used by all ranks
    int input, n, local_n;
    double *X, *local_X, *local_D;
    double avg, global_max, local_max, local_sum, local_var_sum, diff;
    int safe_free= 0;

    while(1){

        //I/O only on rank 0
        if(proc_rank== 0){

            printf("\n=== Menu ===\n");
            printf("1 to enter data - 2 to exit\n");
            scanf("%d", &input);

            //Sending user input to other ranks
            for(int r= 1; r< total_ranks; r++){
                MPI_Send(&input, 1, MPI_INT, r, 8, MPI_COMM_WORLD);
            }

            if(input== 1){
                printf("Enter vector size:\n");
                scanf("%d", &n);
                X= (double*)malloc(n*sizeof(double));

                for(int i= 0; i< n; i++){
                    printf("Enter element %d:\n", i+1);
                    scanf("%lf", &X[i]);
                }

                local_n= n/total_ranks;
                int mod= n%total_ranks;

                //If n%p≠0, rank 0 will take +1 element of the vector
                if(mod> 0) local_n++;
                local_X= (double*)malloc(local_n*sizeof(double));
                for(int i= 0; i< local_n; i++){
                    local_X[i]= X[i];
                }

                //The following ranks will also take on +1 element
                //up to the first rank for which rank > (n%p)-1 applies
                //The rest return to taking on local_n elements.
                int flag= 1;
                int iterated= local_n;
                for(int r= 1; r< total_ranks; r++){
                    if(mod> 0 && r> mod-1 && flag== 1){
                        local_n--;
                        flag= 0;
                    }
                    //Sending the length of the vector corresponding to each process
                    MPI_Send(&local_n, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
                    //Sending the vector corresponding to each process
                    MPI_Send(&X[iterated], local_n, MPI_DOUBLE, r, 1, MPI_COMM_WORLD);
                    iterated+= local_n;
                }
                if(mod> 0) local_n++;
            }
            else if(input== 2) break;
            else{
                printf("Invalid choice\n");
                continue;
            }
        }
        else{
            //Receiving user input
            MPI_Recv(&input, 1, MPI_INT, 0, 8, MPI_COMM_WORLD, &status);

            if(input== 1){
                //Receiving corresponding length
                MPI_Recv(&local_n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
                local_X= (double*)malloc(local_n*sizeof(double));
                //Receiving corresponding vector
                MPI_Recv(local_X, local_n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
            }
            else if(input== 2) break;
            else continue;
        }

        //Local calculation of sum & max for all ranks
        local_sum= 0.0;
        local_max= local_X[0];
        for(int i= 0; i< local_n; i++){
            local_sum+= local_X[i];
            if(local_X[i]> local_max)
                local_max= local_X[i];
        }

        //Sending results back to rank 0
        if(proc_rank> 0){
            MPI_Send(&local_sum, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
            MPI_Send(&local_max, 1, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
        }
        else{
            double global_sum= local_sum;
            double recv_sum, recv_max;
            global_max= local_max;

            //Receiving results from r0 and calculating global average & max
            for(int r= 1; r< total_ranks; r++){
                MPI_Recv(&recv_sum, 1, MPI_DOUBLE, r, 2, MPI_COMM_WORLD, &status);
                MPI_Recv(&recv_max, 1, MPI_DOUBLE, r, 3, MPI_COMM_WORLD, &status);
                global_sum+= recv_sum;
                if(recv_max> global_max)
                    global_max= recv_max;
            }

            avg= global_sum/n;

            //Sending average for local dispersion calculation
            //Sending global max for the calculation of the elements δi of the new vector Δ
            for(int r= 1; r< total_ranks; r++){
                MPI_Send(&avg, 1, MPI_DOUBLE, r, 4, MPI_COMM_WORLD);
                MPI_Send(&global_max, 1, MPI_DOUBLE, r, 6, MPI_COMM_WORLD);
            }
        }
        if(proc_rank> 0){ //Other ranks receiving
            MPI_Recv(&avg, 1, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD, &status);
            MPI_Recv(&global_max, 1, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD, &status);
        }

        //((Xi - μ)^2 calculation from all ranks
        local_var_sum= 0;
        for(int i= 0; i< local_n; i++){
            diff= local_X[i]- avg;
            local_var_sum+= diff*diff;
        }

        //Sending local dispersion to r0 and calculating global dispersion
        if(proc_rank> 0){
            MPI_Send(&local_var_sum, 1, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD);
        }
        else{
            double global_var_sum= local_var_sum;
            double recv_var, var_final;

            for(int r= 1; r< total_ranks; r++){
                MPI_Recv(&recv_var, 1, MPI_DOUBLE, r, 5, MPI_COMM_WORLD, &status);
                global_var_sum+= recv_var;
            }

            var_final= global_var_sum/n;

            //r0 printing results
            printf("\n=== Results ===\n");
            printf("Average μ= %.2f\n", avg);
            printf("Maximum m= %.2f\n", global_max);
            printf("Dispersion var= %.2f\n", var_final);
        }

        //δi calculation from all ranks
        local_D= (double*)malloc(local_n*sizeof(double));
        for(int i= 0; i< local_n; i++){
            local_D[i]= (local_X[i]- global_max) * (local_X[i]- global_max); //(xi-m)^2
        }
        
        //Sending δi to r0
        if(proc_rank> 0){
            MPI_Send(local_D, local_n, MPI_DOUBLE, 0, 7, MPI_COMM_WORLD);
            free(local_D);
        }
        else{ //r0 receiving, calculating and printing
            double *D= (double*)malloc(n*sizeof(double));
            
            int mod= n%total_ranks;
            int flag= 1;
            int iterated= local_n;

            for(int i= 0; i< local_n; i++){
                D[i]= local_D[i];
            }
            free(local_D);

            for(int r= 1; r< total_ranks; r++){
                if(mod> 0 && r> mod-1 && flag== 1){
                    local_n--;
                    flag= 0;
                }
                MPI_Recv(&D[iterated], local_n, MPI_DOUBLE, r, 7, MPI_COMM_WORLD, &status);
                iterated+= local_n;
            }
            printf("New vector Δ= {");
            for(int i= 0; i< n; i++){
                if(i< n-1) printf("%.2f, ", D[i]);
                else printf("%.2f}\n", D[i]);
            }

            free(D);
        }
    }

    if(safe_free== 1){
        free(local_X);
        if(proc_rank== 0) free(X);
    }
    MPI_Finalize();
    return 0;
}
