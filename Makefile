BINS=sgemm_mkl sgemm_cublas

all: $(BINS)

sgemm_mkl: sgemm_mkl.c
	icc -mkl $< -o $@

sgemm_cublas: sgemm_cublas.c
	nvcc -arch=sm_52 $< -o $@ -lcublas

runall: $(BINS)
	COMPUTE_PROFILE=1 ./sgemm_cublas && cat cuda_profile_0.log

clean:
	rm -rf $(BINS) *.log
