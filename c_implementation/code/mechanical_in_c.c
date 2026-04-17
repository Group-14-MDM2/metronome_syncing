#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

/*Types and their constructors*/

typedef struct {
   int n_oscillators;
   float m;
   float M;
   float epsilon;
   float threshold_r;
   float* lengths_ptr;
   float* q_ptr;
   float* dqdt_ptr;
} Model_params;

Model_params build_model_params(int n_oscillators,
                              float m,
                              float M,
                              float epsilon,
                              float threshold_r,
                              float* lengths_ptr,
                              float* q_ptr, 
                              float* dqdt_ptr){
   /*Builder Function for the Model_params struct*/
   Model_params params;
   params.n_oscillators = n_oscillators;
   params.m = m;
   params.M = M;
   params.epsilon = epsilon;
   params.threshold_r = threshold_r;
   params.lengths_ptr = lengths_ptr;
   params.q_ptr = q_ptr;
   params.dqdt_ptr = dqdt_ptr;
   return params;
}

typedef struct{
   float t0;
   float tf;
   int n_steps;
} Run_conditions;

Run_conditions build_run_conds(float t0, float tf, int n_steps){
   /*Builder Function for the Initial_conditions struct*/
   Run_conditions run_conds;
   run_conds.t0 = t0;
   run_conds.tf = tf;
   run_conds.n_steps = n_steps;
   return run_conds;
}

typedef struct {
   Model_params params;
   int coherence_steps;
   float coherence_time;
} Result;

Result build_result(Model_params params, int coherence_steps, float coherence_time){
   /*Builder function for the result struct*/
   Result result;
   result.coherence_steps = coherence_steps;
   result.params = params;
   result.coherence_time = coherence_time;
   return result;
}

/*Running the model*/

float get_coherence(float* q, int n_oscillators){

   // gets the coherence of the oscillators
   complex coherence = 0;
   for(int i = 0; i < n_oscillators - 1; i++){
      coherence += cexpf(I * q[i]);
   }
   coherence /= (n_oscillators - 1);
   return crealf(coherence);
}

Result RK4(Model_params params, Run_conditions run_conditions){
   /* Runs the model using RK4 until either:
               coherence has been reached
               the model has run for n_steps */

   float t = 0;
   float h = (run_conditions.tf - run_conditions.t0) / run_conditions.n_steps;
   for (int i = 0; i <= run_conditions.n_steps; i++){
      // one iteration of the model
      float* q1 = RHS_q();
      float* q2 = RHS_q();
      float* q3 = RHS_q();
      float* q4 = RHS_q();


      // early return if the coherence is above the threshold
      if (get_coherence(params.q_ptr, params.n_oscillators) >= params.threshold_r){
         return build_result(params, i++, t);
      }

      t += h;
   }

   return build_result(params, run_conditions.n_steps, run_conditions.tf);
}


/*Building the models*/

Result* run_models(int n_parameters, Model_params* params_list, Run_conditions run_conditions){

   // where the results are stored in memory
   Result* results = malloc(sizeof(Result) * n_parameters);

   // loops through the model for each set of parameters
   for(int i = 0; i < n_parameters; i++){

      // dereferencing the specific instance of the model parameters
      Model_params params = params_list[i];


      // runs the model until coherence and adds it to the block of memory
      results[i] = RK4(params, run_conditions);


   }

   free(params_list);
   return results;
}


/*The main function*/

int main(int argc, char *argv[]){
   
   if (sizeof(argc) / sizeof(int) != 2){
      printf("Expected One Input: the 'output.csv' where the results should be stored\n");
      return 1;
   }

   // where to write the results to
   char file_path[] = argv[1];

   // Declare all the starting conditions
   Run_conditions run_conditions = build_run_conds(0, 10, 2000);
   
   // Declare the model parameters
   float lengths[3] = {1, 1, 1};
   float q[3] = {0.2, 1.5, 0};
   float dqdt[3] = {0, 0, 0};

   Model_params params[] = {build_model_params(3, 0.15, 1, -0.2, 0.9, &lengths, &q, &dqdt)};


   // Build and runs the simulations

   Result* results = run_models(sizeof(params)/sizeof(Model_params), &params, run_conditions);

   // outputs the results to a given file


   free(results);
   return 0;
}