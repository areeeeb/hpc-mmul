const char *dgemm_desc = "Basic implementation, three-loop dgemm.";

/*
 * This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in row-major format.
 * On exit, A and B maintain their input values.
 */
void square_dgemm(int n, double *A, double *B, double *C)
{
   // Basic matrix multiplication: C = C + A * B
   for (int row = 0; row < n; row++)
   {
      for (int col = 0; col < n; col++)
      {
         for (int inner = 0; inner < n; inner++)
         {
            C[row * n + col] += A[row * n + inner] * B[inner * n + col];
         }
      }
   }
}
