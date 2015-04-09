__kernel myKernel(__global int* ptr) {
  int g = get_group_id(0);
  int s = get_group_size(0);
  int t = get_local_id(0);
  int tid = g * s + t;
  ptr[tid] = tid;
}
