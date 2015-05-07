__kernel void sgemv(
                    __global float * y,
                    const __global float * A,
                    const __global float * x,
                    float alpha,
                    float beta,
                    int nRows,
                    int nCols
                    )
{
  int r = get_global_id(0);
  float result = 0.f;
  for (int c = 0; c < nCols; c++) {
    result += A[nRows*c+r]*x[c];
  }
  y[r] = alpha*result + beta*y[r];
}

