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

// void update_planets(Planet* in, Planet* out) {
//    #pragma omp parallel for
//    for (int i = 0; i < nplanets; i++) {
//       out[i] = in[i];
//    }
//    #pragma omp parallel for
//    // for nplanets > #threads.
//    for (int i = 0; i < nplanets; i++) {
//       for (int j = 0; j < nplanets; j++) {
//          double dx = in[j].x - in[i].x;
//          double dy = in[j].y - in[i].y;
//          double distSqr = dx * dx + dy * dy + 0.0001;  // prevent divide-by-zero
//          double invDist = in[i].mass * in[j].mass / sqrt(distSqr);
//          double invDist3 = invDist * invDist * invDist;
//          out[i].x += dx * invDist3;
//          out[i].y += dy * invDist3;
//       }
//       out[i].x += dt * out[i].vx;
//       out[i].y += dt * out[i].vy;
//    }
// }

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
         double distSqr = dx * dx + dy * dy + 0.0001;  // prevent divide-by-zero
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


// #include <stdio.h>
// #include <stdlib.h>
// #include <math.h>
// #include <omp.h>
// #include <sys/time.h>
// #include <iostream>

// float tdiff(struct timeval *start, struct timeval *end) {
//   return (end->tv_sec-start->tv_sec) + 1e-6*(end->tv_usec-start->tv_usec);
// }

// struct Planet {
//    double mass;
//    double x;
//    double y;
//    double vx;
//    double vy;
// };

// unsigned long long seed = 100;

// unsigned long long randomU64() {
//   seed ^= (seed << 21);
//   seed ^= (seed >> 35);
//   seed ^= (seed << 4);
//   return seed;
// }

// double randomDouble()
// {
//    unsigned long long next = randomU64();
//    next >>= (64 - 26);
//    unsigned long long next2 = randomU64();
//    next2 >>= (64 - 26);
//    return ((next << 27) + next2) / (double)(1LL << 53);
// }

// int nplanets;
// int timesteps;
// double dt;
// double G;
// class QuadTree {
//    private:
//       struct Node {
//          double minX, maxX, minY, maxY;
//          double mass;         
//          double comX, comY;
//          int planetIndex;         
//          Node* children[4];
//          Node(double minX_, double maxX_, double minY_, double maxY_)
//             : minX(minX_), maxX(maxX_), minY(minY_), maxY(maxY_),
//               mass(0.0), comX(0.0), comY(0.0), planetIndex(-1)
//          {
//             for (int i = 0; i < 4; i++)
//                children[i] = nullptr;
//          }
   
//          ~Node() {
//             for (int i = 0; i < 4; i++)
//                if (children[i])
//                   delete children[i];
//          }
//       };
   
//       Node* root;  
//       double THETA;   
   
//       // Determine which quadrant (child index) the point (x, y) belongs to within node.
//       int getChildIndex(Node* node, double x, double y) const {
//          double midX = 0.5 * (node->minX + node->maxX);
//          double midY = 0.5 * (node->minY + node->maxY);
//          bool top   = (y > midY);
//          bool right = (x > midX);
//          if (top && !right)  return 0;
//          if (top && right)   return 1;
//          if (!top && !right) return 2;
//          if (!top && right)  return 3; 
//          return -1;
//       }
//       void insert(Node* node, Planet* planets, int i) {
//          double x = planets[i].x;
//          double y = planets[i].y;
//          double m = planets[i].mass;
   
//          // Check for degenerate node (if bounding box is too small, don't subdivide further)
//          double nodeWidth  = node->maxX - node->minX;
//          double nodeHeight = node->maxY - node->minY;
//          const double epsilon = 1;
//          if (nodeWidth < epsilon || nodeHeight < epsilon) {
//             double oldMass = node->mass;
//             node->mass = oldMass + m;
//             if (node->mass > 1e-15) {
//                node->comX = (node->comX * oldMass + x * m) / node->mass;
//                node->comY = (node->comY * oldMass + y * m) / node->mass;
//             }
//             return;
//          }
   
//          // If node is empty, store this planet.
//          if (node->planetIndex == -1 && node->mass == 0.0) {
//             node->planetIndex = i;
//             node->mass = m;
//             node->comX = x;
//             node->comY = y;
//             return;
//          }
   
//          // If node is a leaf with one planet, subdivide and reinsert the existing planet.
//          if (node->children[0] == nullptr && node->planetIndex != -1) {
//             double midX = 0.5 * (node->minX + node->maxX);
//             double midY = 0.5 * (node->minY + node->maxY);
//             node->children[0] = new Node(node->minX, midX, midY, node->maxY); // NW
//             node->children[1] = new Node(midX, node->maxX, midY, node->maxY); // NE
//             node->children[2] = new Node(node->minX, midX, node->minY, midY); // SW
//             node->children[3] = new Node(midX, node->maxX, node->minY, midY); // SE
   
//             int oldIndex = node->planetIndex;
//             int cidx = getChildIndex(node, planets[oldIndex].x, planets[oldIndex].y);
//             insert(node->children[cidx], planets, oldIndex);
//             node->planetIndex = -1;  
//          }
   
//          // Now insert the new planet into the appropriate child.
//          if (node->children[0] != nullptr) {
//             int cidx = getChildIndex(node, x, y);
//             insert(node->children[cidx], planets, i);
//          }
   
//          // Update this node's mass and center of mass.
//          double oldMass = node->mass;
//          double newMass = oldMass + m;
//          node->mass = newMass;
//          if (newMass > 1e-15) {
//             node->comX = (node->comX * oldMass + x * m) / newMass;
//             node->comY = (node->comY * oldMass + y * m) / newMass;
//          }
//       }
   
   
//       // Recursively compute the force on planet i from the subtree rooted at node.
//       void computeForce(int i, Node* node, Planet* planets, double &fx, double &fy) const {
//          if (node == nullptr || node->mass < 1e-15)
//             return;

//          double xi = planets[i].x;
//          double yi = planets[i].y;
//          double dx = node->comX - xi;
//          double dy = node->comY - yi;
//          double distSqr = dx * dx + dy * dy + 1e-4;

//          if (node->children[0] == nullptr && node->planetIndex != -1) {
//             int j = node->planetIndex;
//             if (j != i) {
//                double invDist = (planets[i].mass * planets[j].mass) / sqrt(distSqr);
//                double invDist3 = invDist * invDist * invDist;
//                fx += dx * invDist3;
//                fy += dy * invDist3;
//             }
//             return;
//          }
   
//          double r = sqrt(distSqr);
//          double size = node->maxX - node->minX;
//          if ((size / r) < THETA) {
//             double invDist = (planets[i].mass * node->mass) / r;
//             double invDist3 = invDist * invDist * invDist;
//             fx += dx * invDist3;
//             fy += dy * invDist3;
//          } else {
//             // Otherwise, recurse into children.
//             for (int c = 0; c < 4; c++) {
//                if (node->children[c] != nullptr)
//                   computeForce(i, node->children[c], planets, fx, fy);
//             }
//          }
//       }
   
//    public:
//       QuadTree(Planet* planets, int n, double theta = 0.5) : THETA(theta) {
//          double minX = planets[0].x, maxX = planets[0].x;
//          double minY = planets[0].y, maxY = planets[0].y;
//          for (int i = 1; i < n; i++) {
//             if (planets[i].x < minX) minX = planets[i].x;
//             if (planets[i].x > maxX) maxX = planets[i].x;
//             if (planets[i].y < minY) minY = planets[i].y;
//             if (planets[i].y > maxY) maxY = planets[i].y;
//          }
//          double dx = maxX - minX;
//          double dy = maxY - minY;
//          double pad = 1e-5;
//          minX -= pad; maxX += pad;
//          minY -= pad; maxY += pad;
//          if (dx < 1e-10) { minX -= 0.5; maxX += 0.5; }
//          if (dy < 1e-10) { minY -= 0.5; maxY += 0.5; }
   
//          root = new Node(minX, maxX, minY, maxY);
//          for (int i = 0; i < n; i++)
//             insert(root, planets, i);
//       }
//       ~QuadTree() {
//          delete root;
//       }

//       void computeForceOnPlanet(int i, Planet* planets, double &fx, double &fy) const {
//          fx = 0.0;
//          fy = 0.0;
//          computeForce(i, root, planets, fx, fy);
//       }
//    };
   
// void update_planets(Planet* in, Planet* out) {
//    // Build the quadtree from the current 'in' array.
//    QuadTree tree(in, nplanets, 0.5);
//    #pragma omp parallel for schedule(static)
//    for (int i = 0; i < nplanets; i++) {
//       double xi  = in[i].x;
//       double yi  = in[i].y;
//       double vxi = in[i].vx;
//       double vyi = in[i].vy;
//       double mi  = in[i].mass;

//       double fx = 0.0, fy = 0.0;
//       tree.computeForceOnPlanet(i, in, fx, fy);

//       out[i].x = xi + fx + dt * vxi;
//       out[i].y = yi + fy + dt * vyi;
//       out[i].vx = vxi;
//       out[i].vy = vyi;
//       out[i].mass = mi;
//    }
// }


// int main(int argc, const char** argv){
//    if (argc < 2) {
//       printf("Usage: %s <nplanets> <timesteps>\n", argv[0]);
//       return 1;
//    }
//    nplanets = atoi(argv[1]);
//    timesteps = atoi(argv[2]);
//    dt = 0.1;
//    G = 6.6743;

//    Planet* planets = (Planet*)malloc(sizeof(Planet) * nplanets);
//    Planet* buffer = (Planet*)malloc(sizeof(Planet) * nplanets);

//    #pragma omp parallel for
//    for (int i = 0; i < nplanets; i++) {
//       planets[i].mass = randomDouble() + 0.1;
//       planets[i].x = randomDouble() * 100 - 50;
//       planets[i].y = randomDouble() * 100 - 50;
//       planets[i].vx = randomDouble() * 5 - 2.5;
//       planets[i].vy = randomDouble() * 5 - 2.5;
//    }

//    struct timeval start, end;
//    gettimeofday(&start, NULL);
//    for (int i = 0; i < timesteps; i++) {
//       update_planets(planets, buffer);
//       Planet* temp = planets;
//       planets = buffer;
//       buffer  = temp;
//       if (i % 10000 == 0) {
//          printf("Timestep %d, location %f %f\n",
//                 i / 10000, planets[nplanets-1].x, planets[nplanets-1].y);
//       }
//    }
//    gettimeofday(&end, NULL);
//    printf("Total time to run simulation %0.6f seconds, final location %f %f\n", tdiff(&start, &end), planets[nplanets-1].x, planets[nplanets-1].y);
//    free(planets);
//    free(buffer);
//    return 0;   
// }
