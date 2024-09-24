const char *dgemm_desc = "Blocked dgemm with copy optimization.";

#include <vector>
#include <cstring> // for memcpy

// Helper function to perform matrix multiplication on blocks
void multiply_blocks(int block_size, double *A_block, double *B_block, double *C_block, int n)
{
   for (int i = 0; i < block_size; ++i)
   {
      for (int j = 0; j < block_size; ++j)
      {
         double sum = 0.0;
         for (int k = 0; k < block_size; ++k)
         {
            sum += A_block[i * block_size + k] * B_block[k * block_size + j];
         }
         C_block[i * n + j] += sum;
      }
   }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in row-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm_blocked(int n, int block_size, double *A, double *B, double *C)
{
   // Calculate number of blocks
   int Nb = n / block_size;

   // Allocate memory for block copies
   std::vector<double> A_block(block_size * block_size);
   std::vector<double> B_block(block_size * block_size);
   std::vector<double> C_block(block_size * block_size);

   // Iterate over blocks
   for (int i = 0; i < Nb; ++i)
   {
      for (int j = 0; j < Nb; ++j)
      {
         // Read/copy block C[i,j] into cache
         for (int x = 0; x < block_size; ++x)
         {
            memcpy(&C_block[x * block_size],
                   &C[(i * block_size + x) * n + j * block_size],
                   block_size * sizeof(double));
         }

         for (int k = 0; k < Nb; ++k)
         {
            // Read/copy block A[i,k] into cache
            for (int x = 0; x < block_size; ++x)
            {
               memcpy(&A_block[x * block_size],
                      &A[(i * block_size + x) * n + k * block_size],
                      block_size * sizeof(double));
            }

            // Read/copy block B[k,j] into cache
            for (int x = 0; x < block_size; ++x)
            {
               memcpy(&B_block[x * block_size],
                      &B[(k * block_size + x) * n + j * block_size],
                      block_size * sizeof(double));
            }

            // C[i,j] += A[i,k] * B[k,j] (matrix multiplication on blocks)
            for (int x = 0; x < block_size; ++x)
            {
               for (int y = 0; y < block_size; ++y)
               {
                  for (int z = 0; z < block_size; ++z)
                  {
                     C_block[x * block_size + y] +=
                         A_block[x * block_size + z] * B_block[z * block_size + y];
                  }
               }
            }
         }

         // Write/copy block C[i,j] back to memory
         for (int x = 0; x < block_size; ++x)
         {
            memcpy(&C[(i * block_size + x) * n + j * block_size],
                   &C_block[x * block_size],
                   block_size * sizeof(double));
         }
      }
   }
}
