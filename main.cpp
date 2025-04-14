#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>

static unsigned long long seed = 100ULL;

static unsigned long long randomU64() {
    seed ^= (seed << 21);
    seed ^= (seed >> 35);
    seed ^= (seed << 4);
    return seed;
}
static double randomDouble() {
    unsigned long long a = randomU64() >> (64 - 26);
    unsigned long long b = randomU64() >> (64 - 26);
    return ((a << 27) + b) / (double)(1ULL << 53);
}

static float tdiff(struct timeval* start, struct timeval* end) {
    return (end->tv_sec - start->tv_sec) + 1e-6f*(end->tv_usec - start->tv_usec);
}

static double* mass_in; 
static double* x_in;    
static double* y_in;      
static double* vx_in;
static double* vy_in;   

static double* mass_out;
static double* x_out;
static double* y_out;
static double* vx_out;
static double* vy_out;

static int nplanets;
static int timesteps;
static double dt = 0.01;     
static double G  = 6.6743;  

void update_planets_SoA(int n)
{
    #pragma omp parallel for
    for(int i=0; i<n; i++){
        double xi   = x_in[i];
        double yi   = y_in[i];
        double vxi  = vx_in[i];
        double vyi  = vy_in[i];
        double mi   = mass_in[i];
        double fx   = 0.0;
        double fy   = 0.0;


        #pragma omp simd reduction(+:fx,fy)
        for(int j=0; j<n; j++){
            if(j == i) continue;
            double dx = x_in[j] - xi;
            double dy = y_in[j] - yi;
            double distSqr = dx*dx + dy*dy + 1e-5; 
            double dist = sqrt(distSqr);
            double F    = G*(mi*mass_in[j]) / distSqr;
            fx += F * (dx/dist);
            fy += F * (dy/dist);
        }
        double ax = fx / mi;
        double ay = fy / mi;
        double newVx = vxi + dt*ax;
        double newVy = vyi + dt*ay;
        double newX  = xi  + dt*newVx;
        double newY  = yi  + dt*newVy;

        mass_out[i] = mi;  
        vx_out[i]   = newVx;
        vy_out[i]   = newVy;
        x_out[i]    = newX;
        y_out[i]    = newY;
    }
}

int main(int argc, char** argv){
    if(argc < 3){
        printf("Usage: %s <nplanets> <timesteps>\n", argv[0]);
        return 1;
    }
    nplanets = atoi(argv[1]);
    timesteps= atoi(argv[2]);

    mass_in = (double*)aligned_alloc(64, nplanets*sizeof(double));
    x_in    = (double*)aligned_alloc(64, nplanets*sizeof(double));
    y_in    = (double*)aligned_alloc(64, nplanets*sizeof(double));
    vx_in   = (double*)aligned_alloc(64, nplanets*sizeof(double));
    vy_in   = (double*)aligned_alloc(64, nplanets*sizeof(double));

    mass_out= (double*)aligned_alloc(64, nplanets*sizeof(double));
    x_out   = (double*)aligned_alloc(64, nplanets*sizeof(double));
    y_out   = (double*)aligned_alloc(64, nplanets*sizeof(double));
    vx_out  = (double*)aligned_alloc(64, nplanets*sizeof(double));
    vy_out  = (double*)aligned_alloc(64, nplanets*sizeof(double));

    #pragma omp parallel for
    for(int i=0; i<nplanets; i++){
        mass_in[i] = randomDouble() + 0.1;
        x_in[i]    = randomDouble()*100.0 - 50.0;
        y_in[i]    = randomDouble()*100.0 - 50.0;
        vx_in[i]   = randomDouble()* 5.0  - 2.5;
        vy_in[i]   = randomDouble()* 5.0  - 2.5;
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    for(int step=0; step<timesteps; step++){
        update_planets_SoA(nplanets);
        double* tmp;
        tmp = mass_in; mass_in = mass_out; mass_out = tmp;
        tmp = x_in;    x_in    = x_out;    x_out    = tmp;
        tmp = y_in;    y_in    = y_out;    y_out    = tmp;
        tmp = vx_in;   vx_in   = vx_out;   vx_out   = tmp;
        tmp = vy_in;   vy_in   = vy_out;   vy_out   = tmp;

      //   if(step % 1000 == 0){
      //       printf("Step %d: last planet at (%.5f, %.5f)\n",
      //              step, x_in[nplanets-1], y_in[nplanets-1]);
      //   }
    }

    gettimeofday(&end, NULL);
    float elapsed = tdiff(&start, &end);
    printf("Total time: %.6f s\n", elapsed);
    printf("Final location: (%.5f, %.5f)\n", x_in[nplanets-1], y_in[nplanets-1]);

    free(mass_in);free(x_in);free(y_in);free(vx_in);free(vy_in);
    free(mass_out);free(x_out);free(y_out);free(vx_out);free(vy_out);
    return 0;
}
