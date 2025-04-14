#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <omp.h>
#include <sys/time.h>

float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec-start->tv_sec) + 1e-6*(end->tv_usec-start->tv_usec);
}

struct Planet {
   double mass;
   double x;
   double y;
   double vx;
   double vy;
};

unsigned long long seed = 100;

unsigned long long randomU64() {
  seed ^= (seed << 21);
  seed ^= (seed >> 35);
  seed ^= (seed << 4);
  return seed;
}

double randomDouble()
{
   unsigned long long next = randomU64();
   next >>= (64 - 26);
   unsigned long long next2 = randomU64();
   next2 >>= (64 - 26);
   return ((next << 27) + next2) / (double)(1LL << 53);
}

int nplanets;
int timesteps;
double dt;
double G;

void update_planets(Planet* in, Planet* out) {
   #pragma omp parallel for schedule(static)
   for (int i = 0; i < nplanets; i++) {
      double xi = in[i].x;
      double yi = in[i].y;
      double vxi = in[i].vx;
      double vyi = in[i].vy;
      double mi = in[i].mass;
      double fx = 0.0, fy = 0.0;

      #pragma omp simd reduction(+:fx,fy)
      for (int j = 0; j < nplanets; j++) {
         double dx = in[j].x - xi;
         double dy = in[j].y - yi;
         double distSqr = dx * dx + dy * dy + 0.0001; 
         double invDist = (mi * in[j].mass) / sqrt(distSqr);
         double invDist3 = invDist * invDist * invDist;
         fx += dx * invDist3;
         fy += dy * invDist3;
      }
      out[i].x = xi + fx + dt * vxi;
      out[i].y = yi + fy + dt * vyi;
      out[i].mass = mi;
      out[i].vx = vxi;
      out[i].vy = vyi;
   }
}


int main(int argc, const char** argv){
   if (argc < 2) {
      printf("Usage: %s <nplanets> <timesteps>\n", argv[0]);
      return 1;
   }
   nplanets = atoi(argv[1]);
   timesteps = atoi(argv[2]);
   dt = 0.1;
   G = 6.6743;

   Planet* planets = (Planet*)malloc(sizeof(Planet) * nplanets);
   Planet* buffer = (Planet*)malloc(sizeof(Planet) * nplanets);
   #pragma omp parallel
   for (int i=0; i<nplanets; i++) {
      planets[i].mass = randomDouble() + 0.1;
      planets[i].x = randomDouble() * 100 - 50;
      planets[i].y = randomDouble() * 100 - 50;
      planets[i].vx = randomDouble() * 5 - 2.5;
      planets[i].vy = randomDouble() * 5 - 2.5;
   }

   struct timeval start, end;
   gettimeofday(&start, NULL);
   for (int i=0; i<timesteps; i++) {
      update_planets(planets, buffer);
      Planet* temp = planets;
      planets = buffer;
      buffer = temp;
      if (i % 10000 == 0) {
         printf("Timestep %d, location %f %f\n", i/10000, planets[nplanets-1].x, planets[nplanets-1].y);
      }
   }
   gettimeofday(&end, NULL);
   printf("Total time to run simulation %0.6f seconds, final location %f %f\n", tdiff(&start, &end), planets[nplanets-1].x, planets[nplanets-1].y);
   free(planets);
   free(buffer);
   return 0;   
}
