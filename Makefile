BINS=sgemm

all: $(BINS)


sgemm: sgemm.c
	nvcc -arch=sm_52 $< -o $@ -lcublas

runall: $(BINS)
	COMPUTE_PROFILE=1 ./sgemm 512 512 512 && mv cuda_profile_0.log sgemm_512_512_512_profile.log;
	COMPUTE_PROFILE=1 ./sgemm 1024 1024 1024 && mv cuda_profile_0.log sgemm_1024_1024_1024_profile.log;
	COMPUTE_PROFILE=1 ./sgemm 2048 2048 2048 && mv cuda_profile_0.log sgemm_2048_2048_2048_profile.log;
	COMPUTE_PROFILE=1 ./sgemm 4096 4096 4096 && mv cuda_profile_0.log sgemm_4096_4096_4096_profile.log;
	COMPUTE_PROFILE=1 ./sgemm 8192 8192 8192 && mv cuda_profile_0.log sgemm_8192_8192_8192_profile.log;

clean:
	rm -rf $(BINS) *.log
